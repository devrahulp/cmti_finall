import React from "react";
import "./Loader.css";

function Loader() {
  return (
    <div className="loader" aria-label="Loading">
      <div className="loader-dot" />
      <div className="loader-dot" />
      <div className="loader-dot" />
    </div>
  );
}

export default Loader;
