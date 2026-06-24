import { useState, useRef, useEffect } from 'react'
import { nanoid } from 'nanoid'
import './App.css'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001'
const MAX_TURNS = 5

function readQueryParams() {
  const params = new URLSearchParams(window.location.search)
  let condition = params.get('condition')
  if (condition !== 'emotion_enhanced' && condition !== 'baseline') {
    console.warn(`[chat] missing/invalid condition param "${condition}", defaulting to "baseline"`)
    condition = 'baseline'
  }
  const pid = params.get('pid') || `anon-${nanoid(8)}`
  const event = params.get('event') || ''
  return { condition, pid, event }
}

function App() {
  const [{ condition, pid, event }] = useState(readQueryParams)
  const sessionIdRef = useRef(nanoid())
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [turn, setTurn] = useState(0)
  const [done, setDone] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [opening, setOpening] = useState(false)
  const openerStartedRef = useRef(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, done])

  useEffect(() => {
    if (openerStartedRef.current) return
    if (!event) return
    openerStartedRef.current = true
    setOpening(true)

    const run = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/chat/open`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId: sessionIdRef.current,
            participantId: pid,
            condition,
            eventDescription: event,
          }),
        })
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}))
          throw new Error(errBody.error || `Server returned ${res.status}`)
        }
        const data = await res.json()
        const reply = data.reply || ''
        setMessages([{ role: 'assistant', content: reply }])
      } catch (err) {
        setErrorMsg(err.message || "Couldn't start the conversation. Please refresh.")
      } finally {
        setOpening(false)
      }
    }
    run()
  }, [event, condition, pid])

  const post = (payload) => {
    try {
      window.parent.postMessage(payload, '*')
    } catch (err) {
      console.warn('[chat] postMessage failed', err)
    }
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading || done) return

    const nextTurn = turn + 1
    const userMessage = { role: 'user', content: text }
    const historyForServer = messages.map((m) => ({ role: m.role, content: m.content }))

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setErrorMsg(null)

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: sessionIdRef.current,
          participantId: pid,
          condition,
          turn: nextTurn,
          history: historyForServer,
          userMessage: text,
        }),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.error || `Server returned ${res.status}`)
      }

      const data = await res.json()
      const reply = data.reply || ''
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
      setTurn(nextTurn)

      post({
        type: 'chat_turn',
        turn: nextTurn,
        condition,
        sessionId: sessionIdRef.current,
        participantId: pid,
      })

      if (nextTurn >= MAX_TURNS || data.isFinal) {
        setDone(true)
        const transcript = [...messages, userMessage, { role: 'assistant', content: reply }]
        post({
          type: 'chat_complete',
          sessionId: sessionIdRef.current,
          participantId: pid,
          condition,
          turns: nextTurn,
          transcript,
        })
      }
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong. Please try sending again.')
      setMessages((prev) => prev.slice(0, -1))
      setInput(text)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const remaining = Math.max(0, MAX_TURNS - turn)

  return (
    <div className="chat-app">
      <header className="chat-header">
        <h1>Chat</h1>
        <span className="turn-counter">
          {done ? 'Conversation complete' : `${remaining} message${remaining === 1 ? '' : 's'} remaining`}
        </span>
      </header>
      <div className="messages">
        {messages.length === 0 && !opening && (
          <p className="placeholder">Tell me what's on your mind — I'm here to listen.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            <div className="message-bubble">{msg.content}</div>
          </div>
        ))}
        {(loading || opening) && (
          <div className="message message-assistant">
            <div className="message-bubble typing">...</div>
          </div>
        )}
        {errorMsg && (
          <div className="error-banner">
            {errorMsg}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {done ? (
        <div className="done-card">
          <strong>Conversation complete.</strong>
          <p>Please scroll down and click <em>Next</em> to continue the survey.</p>
        </div>
      ) : (
        <div className="input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            rows={1}
            disabled={loading || opening}
          />
          <button onClick={sendMessage} disabled={loading || opening || !input.trim()}>
            Send
          </button>
        </div>
      )}
    </div>
  )
}

export default App
