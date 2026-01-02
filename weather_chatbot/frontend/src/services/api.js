
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
});

/**
 * Call the /generate endpoint
 * @param {string} userInput
 * @param {string} threadId
 * @param {boolean} stream
 * @returns {Promise<{assistant_reply: string, events?: Array<{node: string, text: string}>}>}
 */
export async function generate(userInput, threadId = "demo-session-1", stream = false) {
  const payload = { user_input: userInput, thread_id: threadId, stream };
  const { data } = await api.post("/generate", payload);
  return data;
}

export default api;
