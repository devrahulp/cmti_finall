export function generateTitle(prompt) {
  if (!prompt) return "New chat";
  const trimmed = prompt.trim().replace(/\s+/g, " ");
  if (trimmed.length <= 30) return trimmed;
  return trimmed.slice(0, 27) + "...";
}
