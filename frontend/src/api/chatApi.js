// All backend / OpenAI integration goes here.
// Right now it returns a mock answer so the UI works out of the box.

export async function sendMessageToBackend(prompt, files, conversationId) {
  // Example for real backend:
  //
  const formData = new FormData();
  formData.append("prompt", prompt);
  formData.append("conversationId", conversationId);
  files.forEach((file) => formData.append("files", file));
  //
  const res = await fetch("https://your-backend.com/api/chat", {
  method: "POST",
  body: formData,
  });
  const data = await res.json();
  return data.answer;

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  await delay(600);

  return `Mock answer for: "${prompt || "(no text, only files)"}"

(Implement your real backend in src/api/chatApi.js and return a string response here.)`;
}
