import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./nocturne-styles.css";
import "./chrome.css";

// Ship the Violet accent (matched to the Mandaflix logo), per the design handoff.
document.documentElement.style.setProperty("--color-accent", "#9b4dff");

createRoot(document.getElementById("root")).render(<App />);
