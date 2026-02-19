import { useState, useRef, useEffect } from 'react'
import OpenAI from 'openai'
import { SYMPATHETIC_SYSTEM_PROMPT } from './prompts'
import './App.css'

const openai = new OpenAI({
  apiKey: import.meta.env.VITE_OPENAI_API_KEY,
  dangerouslyAllowBrowser: true,
})

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMessage = { role: 'user', content: text }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const apiMessages = [
        { role: 'system', content: SYMPATHETIC_SYSTEM_PROMPT },
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        userMessage,
      ]
      const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: apiMessages,
      })
      const reply = completion.choices[0]?.message?.content ?? 'Sorry, I couldn’t respond.'
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message || 'Something went wrong.'}` },
      ])
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

  return (
    <div className="chat-app">
      <header className="chat-header">
        <h1>Sympathetic Chat</h1>
      </header>
      <div className="messages">
        {messages.length === 0 && (
          <p className="placeholder">Say something — I’m here to listen.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            <div className="message-bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="message message-assistant">
            <div className="message-bubble typing">...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          rows={1}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}

export default App
