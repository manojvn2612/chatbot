import React, { useState } from 'react';
import logo from './logo.svg';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim()) {
      // Add user message to the chat
      setMessages([...messages, { text: input, sender: 'user' }]);
      
      try {
        // Send user message to the backend
        const response = await axios.post('http://127.0.0.1:5000/predict', {
          message: input, // Send the user's input
        });
        
        // Add AI response to the chat
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: response.data.response, sender: 'ai' } // Use the response from the server
        ]);
      } catch (error) {
        console.error("Error sending message to the backend:", error);
        // Optionally, you can add an error message to the chat
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: 'Error communicating with the AI.', sender: 'ai' }
        ]);
      }

      setInput(''); // Clear the input
    }
  };

  return (
    <div className="App">
      <aside className="sidemenu">
        <div className="sidemenu-button">New Chat</div>
      </aside>
      <section className="chatbox">
        <div className="chat-log">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.sender === 'user' ? '' : 'chat-message-server'}`}
            >
              <div className="avatar"></div>
              <div className="message">{msg.text}</div>
            </div>
          ))}
        </div>
        <div className="chat-input-holder">
          <form onSubmit={handleSubmit}>
            <textarea
              className="chat_input-textarea"
              placeholder="Start typing..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows="3" // Set the number of visible text lines
            />
            <button type="submit" className="chat_submit-button">
              <img src={require('./assets/message.png')} alt="Submit" className="submit-icon" />
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}

export default App;
