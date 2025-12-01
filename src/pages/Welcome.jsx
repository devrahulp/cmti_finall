import React, { useState } from "react";
import "../styles/Welcome.css";

import CMTILogo from "../assets/images/CMTILogo.jpg";
import DocAILogo from "../assets/images/docai.jpeg";
import finalbg from "../assets/images/finalbg.jpg";
import downloadbg from "../assets/images/download.jpg";



export default function App() {
  const [showCards, setShowCards] = useState(false);

  const handleExplore = () => {
    setShowCards(true);
    setTimeout(() => {
      window.scrollTo({ top: 650, behavior: "smooth" });
    }, 200);
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="header">
        <div className="header-left">
          <img src={DocAILogo} alt="DocAI Logo" className="docai-header-logo" />
        </div>
        <div className="header-right">
          <img src={CMTILogo} alt="CMTI Logo" className="header-logo" />
        </div>
      </header>

      {/* MAIN BLUR CARD */}
      <div
        className="blur-card"
        style={{ backgroundImage: `url(${finalbg})` }}
      >
        <h1 className="welcome">WELCOME TO</h1>
        <h1 className="docai">Doc.ai</h1>

        <p className="subtitle">
          Transforming handwritten documents into accurate, searchable,<br />
          and intelligent digital content.
        </p>

        <button className="explore-btn" onClick={handleExplore}>
          Explore
        </button>
      </div>

      {/* POP-UP CARDS */}
      {showCards && (
        <>
          <div className="cards-section">
            <div className="info-card">
              Doc.AI simplifies workflows by digitizing handwritten notes and
              extracting meaningful content.
            </div>

            <div className="info-card">
              This platform transforms documents into clean, searchable text for
              improved understanding and organization.
            </div>

            <div className="info-card">
              With accurate AI-powered processing, document handling becomes faster
              and more reliable than ever.
            </div>
          </div>

          {/* TRY NOW BELOW CARDS */}
          <div className="try-lower-wrapper">
            <button
              className="try-lower-btn"
              onClick={() => (window.location.href = "/convert")}
            >
              Try Now
            </button>
          </div>
        </>
      )}
    </div>
  );
}
