import React, { useEffect, useRef, useState } from "react";
import { api } from "./api";

const initials = (name) => name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
const fmtRuntime = (m) => (m ? `${Math.floor(m / 60)}h ${m % 60}m` : null);
const fmtRatings = (n) => n.toLocaleString("en-US");
const nice = (key) =>
  key.replace(/_/g, " ").replace(/\b(cf|svd|xgboost|ndcg|fts|mrr)\b/gi, (m) => m.toUpperCase())
    .replace(/\b\w/g, (c) => c.toUpperCase());

function Poster({ movie }) {
  if (movie.poster) return <img src={movie.poster} alt={movie.title} loading="lazy" />;
  return <div className="poster-ph">{movie.title}</div>;
}

function Tile({ movie, onOpen, showRank, grid }) {
  return (
    <div className={"tile" + (grid ? " grid-tile" : "")} onClick={() => onOpen(movie.movieId)}>
      <div className="poster">
        <Poster movie={movie} />
        {showRank && movie.rank != null && <span className="rank-badge">{movie.rank}</span>}
      </div>
      <div className="tile-title">{movie.title}</div>
      <div className="tile-sub">{movie.year}{movie.genres[0] ? ` · ${movie.genres[0]}` : ""}</div>
    </div>
  );
}

function MetaRow({ movie }) {
  const parts = [movie.year, fmtRuntime(movie.runtime), movie.genres.slice(0, 3).join(", ")].filter(Boolean);
  return (
    <div className="meta">
      {parts.map((p, i) => (
        <React.Fragment key={i}>
          {i > 0 && <span className="dot">·</span>}<span>{p}</span>
        </React.Fragment>
      ))}
    </div>
  );
}

function Header({ view, users, user, onUser, onNav, onSearch }) {
  const [open, setOpen] = useState(false);
  return (
    <header className="app-header">
      <button className="brand" onClick={() => onNav("home")}>Mandaflix</button>
      <div className="spacer" />
      <button className={"icon-btn" + (view === "search" ? " active" : "")} title="Search" onClick={onSearch}>
        <i className="ph ph-magnifying-glass" />
      </button>
      <button className={"icon-btn" + (view === "insights" ? " active" : "")} title="Insights" onClick={() => onNav("insights")}>
        <i className="ph ph-chart-bar" />
      </button>
      <div className="profile">
        <button className="profile-trigger" onClick={() => setOpen((o) => !o)}>
          <span className="avatar sm">{user ? initials(user.name) : "?"}</span>
          <span className="name">{user?.name}</span>
          <i className="ph ph-caret-down" />
        </button>
        {open && (
          <div className="profile-panel">
            <div className="panel-label">Switch profile</div>
            {users.map((u) => (
              <button key={u.id} className={"profile-row" + (u.id === user?.id ? " selected" : "")}
                onClick={() => { onUser(u); setOpen(false); }}
                title={`${u.name} — user #${u.id} · ${fmtRatings(u.ratings)} ratings`}>
                <span className="avatar">{initials(u.name)}</span>
                <div>
                  <div className="name">{u.name}</div>
                  <div className="sub">user #{u.id} · {fmtRatings(u.ratings)} ratings</div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </header>
  );
}

function Hero({ movie, user, onOpen }) {
  const bg = movie.backdrop
    ? { backgroundImage: `url(${movie.backdrop})` }
    : { background: "repeating-linear-gradient(135deg,#232532,#232532 14px,#292b31 14px,#292b31 28px)" };
  return (
    <section className="hero">
      <div className="hero-bg" style={bg} />
      <div className="hero-scrim" />
      <div className="hero-fade" />
      <div className="hero-content">
        <div className="kicker">Top pick for {user?.name}</div>
        <h1>{movie.title}</h1>
        <MetaRow movie={movie} />
        {movie.synopsis && <p className="synopsis">{movie.synopsis}</p>}
        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => onOpen(movie.movieId)}><i className="ph ph-info" /> More info</button>
        </div>
      </div>
    </section>
  );
}

function Row({ title, note, movies, onOpen, showRank }) {
  if (!movies.length) return null;
  return (
    <section className="section">
      <div className="row-head"><h2>{title}</h2>{note && <span className="note">{note}</span>}</div>
      <div className="row scroll-x">
        {movies.map((m) => <Tile key={m.movieId} movie={m} onOpen={onOpen} showRank={showRank} />)}
      </div>
    </section>
  );
}

function Home({ recs, likes, genres, genre, onGenre, user, onOpen }) {
  if (!recs.length) return <div className="page"><p className="note">No recommendations.</p></div>;
  const note = (genre !== "All" ? `${genre} · ` : "") + `ranked for ${user?.name}`;
  return (
    <>
      <Hero movie={recs[0]} user={user} onOpen={onOpen} />
      <div className="chips scroll-x" style={{ marginTop: 22.4 }}>
        {genres.map((g) => (
          <button key={g} className={"chip" + (g === genre ? " active" : "")} onClick={() => onGenre(g)}>{g}</button>
        ))}
      </div>
      <Row title="Favourites" note={`liked by ${user?.name}`} movies={likes} onOpen={onOpen} />
      <Row title="For you" note={note} movies={recs.slice(0, 10)} onOpen={onOpen} showRank />
      <Row title="More for you" note={note} movies={recs.slice(10, 20)} onOpen={onOpen} />
    </>
  );
}

function SearchScreen({ query, onQuery, results, popular, onOpen }) {
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); }, []);
  const q = query.trim();
  return (
    <div className="search-page">
      <div className="search-big">
        <i className="ph ph-magnifying-glass" />
        <input ref={ref} value={query} onChange={(e) => onQuery(e.target.value)}
          placeholder="Search titles, genres, people, or describe a plot…" />
      </div>
      {q ? (
        <>
          <div className="search-section-label">{results.length} results for “{q}”</div>
          <div className="grid">{results.map((m) => <Tile key={m.movieId} movie={m} onOpen={onOpen} grid />)}</div>
        </>
      ) : (
        <>
          <div className="search-section-label">Popular</div>
          <div className="grid">{popular.map((m) => <Tile key={m.movieId} movie={m} onOpen={onOpen} grid />)}</div>
        </>
      )}
    </div>
  );
}

function Detail({ movie, similar, onBack, onOpen, onToggleLike }) {
  const bg = movie.backdrop
    ? { backgroundImage: `url(${movie.backdrop})`, backgroundSize: "cover", backgroundPosition: "center 30%" }
    : { background: "repeating-linear-gradient(135deg,#232532,#232532 14px,#292b31 14px,#292b31 28px)" };
  return (
    <>
      <div className="detail-bg">
        <div style={{ position: "absolute", inset: 0, ...bg }} />
        <div className="detail-scrim" />
        <button className="back-btn" onClick={onBack}><i className="ph ph-arrow-left" /> Back</button>
      </div>
      <div className="detail-body">
        <div className="detail-poster"><Poster movie={movie} /></div>
        <div className="detail-text">
          <h1>{movie.title}</h1>
          <MetaRow movie={movie} />
          <div className="detail-actions">
            <button className={"btn " + (movie.liked ? "btn-primary" : "btn-secondary")}
              onClick={() => onToggleLike(movie)}>
              <i className="ph ph-thumbs-up" /> {movie.liked ? "Liked" : "Like"}
            </button>
          </div>
          {movie.synopsis && <p className="detail-synopsis">{movie.synopsis}</p>}
          <div className="credits">
            {movie.director && <div><div className="label">Director</div><div className="value">{movie.director}</div></div>}
            {movie.cast?.length > 0 && <div><div className="label">Cast</div><div className="value">{movie.cast.join(", ")}</div></div>}
            <div><div className="label">Genres</div><div className="value">{movie.genres.join(", ")}</div></div>
          </div>
        </div>
      </div>
      <Row title="Similar movies" movies={similar} onOpen={onOpen} />
      <div style={{ height: 44.8 }} />
    </>
  );
}

function MetricTable({ title, note, rows, cols, bestKey }) {
  const best = rows.reduce((b, r) => (r[1][bestKey] > (b?.[1]?.[bestKey] ?? -1) ? r : b), null);
  return (
    <>
      <h2>{title}</h2>
      <div className="note">{note}</div>
      <table className="admin-table">
        <thead><tr><th>{cols.label}</th>{cols.metrics.map((m) => <th key={m} className="num">{m.header}</th>)}</tr></thead>
        <tbody>
          {rows.map(([key, m]) => (
            <tr key={key} className={key === best?.[0] ? "best" : ""}>
              <td>{nice(key)}{key === best?.[0] ? " ★" : ""}</td>
              {cols.metrics.map((c) => <td key={c.field} className="num">{(m[c.field] ?? 0).toFixed(3)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Insights({ metrics, compare, searchMetrics, goldenMetrics, user }) {
  if (!metrics) return <div className="admin"><p className="note">Loading…</p></div>;
  const rk = Object.values(metrics)[0]?.k ?? 10;
  return (
    <div className="admin">
      <MetricTable
        title="Recommendation models" bestKey={`ndcg@${rk}`}
        note={`What the consumer UI hides: every model tier and its offline metrics (temporal per-user split, relevance = rating ≥ 4). The app serves SVD.`}
        rows={Object.entries(metrics)}
        cols={{ label: "Model", metrics: [
          { header: `Precision@${rk}`, field: `precision@${rk}` },
          { header: `Recall@${rk}`, field: `recall@${rk}` },
          { header: `NDCG@${rk}`, field: `ndcg@${rk}` },
        ] }}
      />
      {searchMetrics && (
        <div style={{ marginTop: 40 }}>
          <MetricTable
            title="Search — hybrid vs its parts" bestKey="mrr"
            note="Synthetic known-item eval (title + plot-snippet queries). FTS leads here — this set is title-heavy and exact titles are its home turf."
            rows={Object.entries(searchMetrics)}
            cols={{ label: "Retriever", metrics: [
              { header: "MRR", field: "mrr" },
              { header: "Recall@10", field: "recall@10" },
              { header: "NDCG@10", field: "ndcg@10" },
            ] }}
          />
        </div>
      )}
      {goldenMetrics && (
        <div style={{ marginTop: 40 }}>
          <MetricTable
            title="Search — golden set (semantic queries)" bestKey="ndcg@10"
            note="Human-labelled queries by plot/vibe (‘a heist with a clever twist’), several relevant each. Here the picture flips: semantic beats FTS, and hybrid is the most robust — best recall and NDCG. The two evals together: FTS for titles, semantic for vibes, hybrid across both."
            rows={Object.entries(goldenMetrics)}
            cols={{ label: "Retriever", metrics: [
              { header: "MRR", field: "mrr" },
              { header: "Recall@10", field: "recall@10" },
              { header: "NDCG@10", field: "ndcg@10" },
            ] }}
          />
        </div>
      )}
      <h2 style={{ marginTop: 40 }}>Per-model top picks — {user?.name} (user #{user?.id})</h2>
      <div className="note">The same user, ranked by each model. Consumer UI never shows these side by side.</div>
      <div className="admin-cols">
        {compare && Object.entries(compare).map(([key, items]) => (
          <div className="admin-col" key={key}>
            <h3>{nice(key)}</h3>
            {items.map((it, i) => (
              <div className="admin-item" key={i}>
                <span className="r">{i + 1}</span><span>{it.title}</span>
                {it.score != null && <span className="s">{it.score.toFixed(2)}</span>}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [users, setUsers] = useState([]);
  const [genresList, setGenres] = useState(["All"]);
  const [user, setUser] = useState(null);
  const [view, setView] = useState("home");
  const [genre, setGenre] = useState("All");
  const [recs, setRecs] = useState([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [popular, setPopular] = useState([]);
  const [movie, setMovie] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [likes, setLikes] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [compare, setCompare] = useState(null);
  const [searchMetrics, setSearchMetrics] = useState(null);
  const [goldenMetrics, setGoldenMetrics] = useState(null);
  const debounce = useRef(null);

  useEffect(() => {
    api.users().then((u) => { setUsers(u); setUser(u[0]); });
    api.genres().then(setGenres);
    api.popular().then(setPopular);
  }, []);

  const refreshUser = (u) => {
    api.recommendations(u.id, { k: 20, genre }).then(setRecs);
    api.userLikes(u.id).then(setLikes);
  };

  useEffect(() => {
    if (user) refreshUser(user);
  }, [user, genre]);

  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    if (!query.trim()) { setResults([]); return; }
    debounce.current = setTimeout(() => api.search(query).then(setResults), 200);
  }, [query]);

  const openMovie = (id) => {
    Promise.all([api.movie(id, user.id), api.similar(id, user.id)]).then(([m, s]) => {
      setMovie(m); setSimilar(s); setView("detail"); window.scrollTo(0, 0);
    });
  };

  const toggleLike = async (m) => {
    await api.like(user.id, m.movieId, !m.liked);
    setMovie({ ...m, liked: !m.liked });   // instant UI feedback
    refreshUser(user);                      // re-rank + refresh favourites
  };

  const nav = (v) => {
    if (v === "insights") {
      setView("insights");
      api.adminMetrics().then(setMetrics);
      api.adminSearchMetrics().then(setSearchMetrics);
      api.adminGoldenMetrics().then(setGoldenMetrics);
      if (user) api.adminCompare(user.id).then(setCompare);
    } else setView("home");
  };

  const selectUser = (u) => { setUser(u); setGenre("All"); setView("home"); };

  useEffect(() => {
    if (view === "insights" && user) api.adminCompare(user.id).then(setCompare);
  }, [user]);

  return (
    <div className="app">
      <Header view={view} users={users} user={user} onUser={selectUser} onNav={nav}
        onSearch={() => setView("search")} />
      <main>
        {view === "home" && (
          <Home recs={recs} likes={likes} genres={genresList} genre={genre}
            onGenre={(g) => { setGenre(g); setView("home"); }} user={user} onOpen={openMovie} />
        )}
        {view === "search" && (
          <SearchScreen query={query} onQuery={setQuery} results={results} popular={popular} onOpen={openMovie} />
        )}
        {view === "detail" && movie && <Detail movie={movie} similar={similar} onBack={() => setView("home")} onOpen={openMovie} onToggleLike={toggleLike} />}
        {view === "insights" && <Insights metrics={metrics} compare={compare} searchMetrics={searchMetrics} goldenMetrics={goldenMetrics} user={user} />}
      </main>
      <footer className="footer">
        <span className="brand">Mandaflix</span>
        <span>An AI movie recommendations project based on the MovieLens dataset.</span>
      </footer>
    </div>
  );
}
