// Thin API client. Paths are same-origin (/api/…); Vite proxies to FastAPI in dev.
const j = (url) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
});

export const api = {
  users: () => j("/api/users"),
  genres: () => j("/api/genres"),
  recommendations: (userId, { k = 20, genre = "All" } = {}) =>
    j(`/api/recommendations?user_id=${userId}&k=${k}&genre=${encodeURIComponent(genre)}`),
  search: (q) => j(`/api/search?q=${encodeURIComponent(q)}`),
  popular: (k = 24) => j(`/api/popular?k=${k}`),
  movie: (id, userId) => j(`/api/movies/${id}${userId != null ? `?user_id=${userId}` : ""}`),
  similar: (id, userId, k = 7) => j(`/api/movies/${id}/similar?user_id=${userId}&k=${k}`),
  userLikes: (userId) => j(`/api/users/${userId}/likes`),
  like: (userId, movieId, liked = true) =>
    fetch("/api/likes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, movie_id: movieId, liked }),
    }).then((r) => r.json()),
  // insights
  adminMetrics: () => j("/api/admin/metrics"),
  adminCompare: (userId, k = 8) => j(`/api/admin/compare?user_id=${userId}&k=${k}`),
  adminSearchMetrics: () => j("/api/admin/search-metrics"),
  adminGoldenMetrics: () => j("/api/admin/golden-metrics"),
};
