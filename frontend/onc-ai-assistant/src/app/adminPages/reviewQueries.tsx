import './adminPanel.css';
import {FormEvent, useState} from "react";

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
const m6: Message = {text: "test another unrated message but long enough to overflow to newline for real this time", rating: 0}

const sampleMessages: Message[] = [m1, m2, m3, m4, m5, m6]
// ------------------------------ //

export default function ReviewQueries() {

    const [queries, setQueries] = useState<Message[]>([]);

    //eventually this will call the messages API endpoint
    const retrieveQueries = (e: FormEvent) => {
        const event = e.target as HTMLFormElement;
        const rate = event.value;
        const retrieved: Message[] = rate == 2 ? sampleMessages : sampleMessages.filter(msg => msg.rating == rate);

        setQueries(retrieved);
    }

    return(
        <div className="module">
        <h2>Review User Feedback & Frequent Queries</h2>
        <div className="frequent-queries">
            <div className="rating-filter">
                <label htmlFor="rating">Select messages to show:</label>
                <select id="rating" name="rating" onChange={(e) => retrieveQueries(e)}>
                    <option value={2}>All Messages</option>
                    <option value={1}>Positive</option>
                    <option value={-1}>Negative</option>
                    <option value={0}>Not Rated</option>
                </select>
            </div>
            <ul>
                {queries.map(message => <li>{message.text}, {message.rating.toString()}</li>)}
            </ul>
        </div>
        </div>
    )
}