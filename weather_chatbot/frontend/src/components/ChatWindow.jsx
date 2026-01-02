
export default function ChatWindow({ messages, events }) {
  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((m, idx) => (
          <div key={idx} className={`msg ${m.role}`}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong>{" "}
            <span>{m.text}</span>
          </div>
        ))}
      </div>

      {events && events.length > 0 && (
        <div className="events">
          <h4>Agent events</h4>
          <ul>
            {events.map((e, i) => (
              <li key={i}>
                <code>[{e.node}]</code> {e.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
