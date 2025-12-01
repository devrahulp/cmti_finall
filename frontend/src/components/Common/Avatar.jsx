import React from "react";
import "./Avatar.css";

function Avatar({ label, small = false }) {
  const initial = label?.[0]?.toUpperCase() || "U";
  return (
    <div className={`avatar-circle ${small ? "small" : ""}`}>
      {initial}
    </div>
  );
}

export default Avatar;
