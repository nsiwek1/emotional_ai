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

// Reads a chat response. Streams Server-Sent Events ("data: {json}\n\n") and
// calls onDelta(accumulatedText) for each token; also handles the plain-JSON
// fallback used when the backend runs with STREAMING=0. Returns the final
// { reply, turn, isFinal }. Throws if the stream sends an error event.
async function consumeChat(res, onDelta) {
  const contentType = res.headers.get('content-type') || ''
  if (!contentType.includes('text/event-stream') || !res.body) {
    const data = await res.json()
    const reply = data.reply || ''
    if (reply) onDelta(reply)
    return { reply, turn: data.turn, isFinal: data.isFinal }
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let full = ''
  let result = { reply: '', turn: undefined, isFinal: false }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const line = buffer.slice(0, sep).replace(/^data:\s?/, '').trim()
      buffer = buffer.slice(sep + 2)
      if (!line) continue
      let msg
      try {
        msg = JSON.parse(line)
      } catch {
        continue
      }
      if (msg.type === 'delta') {
        full += msg.text
        onDelta(full)
      } else if (msg.type === 'done') {
        result = { reply: msg.reply ?? full, turn: msg.turn, isFinal: msg.isFinal }
      } else if (msg.type === 'error') {
        throw new Error(msg.error || 'stream_error')
      }
    }
  }
  if (!result.reply) result.reply = full
  return result
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
        const { reply } = await consumeChat(res, (acc) => {
          setMessages([{ role: 'assistant', content: acc }])
        })
        if (reply) setMessages([{ role: 'assistant', content: reply }])
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

    let started = false
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

      const { reply, isFinal } = await consumeChat(res, (acc) => {
        setMessages((prev) => {
          const copy = prev.slice()
          if (started) {
            copy[copy.length - 1] = { role: 'assistant', content: acc }
          } else {
            started = true
            copy.push({ role: 'assistant', content: acc })
          }
          return copy
        })
      })

      // Normalize the final bubble to the server's trimmed text.
      setMessages((prev) => {
        const copy = prev.slice()
        if (copy.length && copy[copy.length - 1].role === 'assistant') {
          copy[copy.length - 1] = { role: 'assistant', content: reply }
        } else {
          copy.push({ role: 'assistant', content: reply })
        }
        return copy
      })
      setTurn(nextTurn)

      post({
        type: 'chat_turn',
        turn: nextTurn,
        condition,
        sessionId: sessionIdRef.current,
        participantId: pid,
      })

      if (nextTurn >= MAX_TURNS || isFinal) {
        setDone(true)
        const transcript = [...historyForServer, userMessage, { role: 'assistant', content: reply }]
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
      setMessages((prev) => {
        let copy = prev.slice()
        if (started && copy.length && copy[copy.length - 1].role === 'assistant') copy = copy.slice(0, -1)
        if (copy.length && copy[copy.length - 1].role === 'user') copy = copy.slice(0, -1)
        return copy
      })
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
  // Show the typing dots only until the assistant starts streaming its reply.
  const waitingForReply =
    (loading || opening) && (messages.length === 0 || messages[messages.length - 1].role !== 'assistant')

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
        {waitingForReply && (
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
