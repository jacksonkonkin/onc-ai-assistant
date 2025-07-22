import './adminPanel.css';
import {useState} from "react";

type Message = {
    text: String;
    rating: Number;
}

// this whole section will get cleaned up when backend integration
const m1: Message = {text: "test positive message", rating: 1}
const m2: Message = {text: "test negative message", rating: -1}
const m3: Message = {text: "test unrated message", rating: 0}
const m4: Message = {text: "test another positive message", rating: 1}
const m5: Message = {text: "test another negative message", rating: -1}
const m6: Message = {text: "test another unrated message", rating: 0}

const sampleMessages: Message[] = [m1, m2, m3, m4, m5 ,m6]
// ------------------------------ //

export default function ReviewQueries() {

    const [queries, setQueries] = useState<Message[]>([]);

    //eventually this will call the messages API endpoint
    const retrieveQueries = (rate: Number) => {
        const retrieved: Message[] = sampleMessages.filter(msg => msg.rating == rate);
        setQueries(retrieved);
    }


    return(
        <div className="module">
        <h2>Review User Feedback & Frequent Queries</h2>
        <div className="frequent-queries">
            <div className="rating-filter">
                <label htmlFor="rating">Select messages to show:</label>
                <select id="rating" name="rating" onChange={() => retrieveQueries}>
                    <option value={2}>All Messages</option>
                    <option value={1}>Positive</option>
                    <option value={-1}>Negative</option>
                    <option value={0}>Not Rated</option>
                </select>
            </div>
            <ul>
            <li>What is the average temperature in Cambridge Bay in July?</li>
            <li>What is the current temperature in Cambridge Bay?</li>
            <li>Is there a turbidity sensor in Cambridge Bay?</li>
            </ul>
        </div>
        </div>
    )
}