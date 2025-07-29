"use client";

import { useState } from "react";
import { FiThumbsUp, FiThumbsDown } from "react-icons/fi";

interface FeedbackButtonsProps {
  query: string;
  response: string;
  onFeedbackSubmitted?: (rating: string) => void;
}

export default function FeedbackButtons({ query, response, onFeedbackSubmitted }: FeedbackButtonsProps) {
  const [selectedRating, setSelectedRating] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitFeedback = async (rating: string) => {
    if (isSubmitting || selectedRating) return; // Prevent multiple submissions
    
    setIsSubmitting(true);
    
    try {
      const response_api = await fetch("http://localhost:8000/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rating: rating,
          query: query,
          response: response,
          chatID: 0,
          userID: 0
        }),
      });

      if (response_api.ok) {
        setSelectedRating(rating);
        onFeedbackSubmitted?.(rating);
      } else {
        console.error("Failed to submit feedback");
      }
    } catch (error) {
      console.error("Error submitting feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getButtonStyle = (rating: string) => ({
    border: "none",
    background: "none",
    cursor: selectedRating ? "default" : "pointer",
    padding: "8px",
    borderRadius: "6px",
    display: "flex",
    alignItems: "center",
    gap: "4px",
    fontSize: "14px",
    transition: "all 0.2s ease",
    opacity: selectedRating && selectedRating !== rating ? 0.3 : 1,
    backgroundColor: selectedRating === rating ? 
      (rating === "thumbs_up" ? "#e8f5e8" : "#ffeaea") : "transparent",
    color: selectedRating === rating ? 
      (rating === "thumbs_up" ? "#2d5a2d" : "#722f2f") : "#666",
    ...(selectedRating === null && !isSubmitting ? {
      ":hover": {
        backgroundColor: rating === "thumbs_up" ? "#f0f8f0" : "#fff0f0"
      }
    } : {})
  });

  if (selectedRating) {
    return (
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "8px",
        marginTop: "8px",
        fontSize: "14px",
        color: "#666"
      }}>
        <span>
          {selectedRating === "thumbs_up" ? "👍" : "👎"} 
          Thanks for your feedback!
        </span>
      </div>
    );
  }

  return (
    <div style={{
      display: "flex",
      gap: "8px",
      marginTop: "8px",
      alignItems: "center"
    }}>
      <span style={{ fontSize: "14px", color: "#888" }}>Was this helpful?</span>
      
      <button
        onClick={() => submitFeedback("thumbs_up")}
        disabled={isSubmitting || selectedRating !== null}
        style={getButtonStyle("thumbs_up")}
        title="Thumbs up"
      >
        <FiThumbsUp size={16} />
      </button>
      
      <button
        onClick={() => submitFeedback("thumbs_down")}
        disabled={isSubmitting || selectedRating !== null}
        style={getButtonStyle("thumbs_down")}
        title="Thumbs down"
      >
        <FiThumbsDown size={16} />
      </button>
    </div>
  );
}