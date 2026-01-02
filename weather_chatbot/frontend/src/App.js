
import { useState } from "react";
import ChatInput from "./components/ChatInput.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import { generate } from "./services/api.js";
import "./styles.css";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [events, setEvents] = useState([]);
  const [threadId, setThreadId] = useState("demo-session-1");
  const [stream, setStream] = useState(false);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (text) => {
    setLoading(true);
    // Show user message immediately
    setMessages((prev) => [...prev, { role: "user", text }]);
    try {
      const res = await generate(text, threadId, stream);
      const reply = res?.assistant_reply || "(No reply)";
      setMessages((prev) => [...prev, { role: "assistant", text: reply }]);

      if (stream && Array.isArray(res?.events)) {
        setEvents(res.events);
      } else {
        setEvents([]);
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error talking to the server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Weather Chatbot</h1>
        <div className="controls">
          <label>
            Thread ID:
            <input
              value={threadId}
              onChange={(e) => setThreadId(e.target.value)}
              placeholder="demo-session-1"
            />
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
            />
            Stream agent events
          </label>
        </div>
      </header>

      <ChatWindow messages={messages} events={events} />
      <ChatInput onSend={sendMessage} disabled={loading} />
    </div>
  );
}
