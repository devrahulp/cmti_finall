import React from "react";
import "./Button.css";

function Button({ children, variant = "secondary", className = "", ...rest }) {
  const base = "btn";
  const variantClass = variant === "primary" ? "btn-primary" : "btn-secondary";
  return (
    <button className={`${base} ${variantClass} ${className}`.trim()} {...rest}>
      {children}
    </button>
  );
}

export default Button;
