import './adminPanel.css';
import {FormEvent, Key, useState} from "react";

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
const m7: Message = {text: "test another message but long enough to overflow to newline for real this time", rating: -1}

const sampleMessages: Message[] = [m1, m2, m3, m4, m5, m6, m7]
// ------------------------------ //

export default function ReviewQueries() {

    const [queries, setQueries] = useState<Message[]>([]);

    const fetchMessage = async(r: Number) => {
        try {
            const response = await fetch(`https://onc-assistant-822f952329ee.herokuapp.com/api/messages-by-rating?rating=${r}`,
                {
                method: "GET",       
            })

            if (!response.ok) {
                throw new Error("API request failed");
            }

            const messages = await response.json();
            console.log(messages.response)
            // need to format for display still
            return messages.response;
        } catch (error) {
            console.error("Error: ", error);
            return "Error retrieving messages.";
        }
    }

    //eventually this will call the messages API endpoint
    const retrieveQueries = (e: FormEvent) => {
        const event = e.target as HTMLFormElement;
        const rate = event.value;
        

        const retrieved: Message[] = rate == 2 ? sampleMessages : sampleMessages.filter(msg => msg.rating == rate);
        

        // const retrieved: Message[] = fetchMessage(0).toArray();
        // fetchMessage(rate)

        

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
            {/* <span className="query-display"> */}
                <ul className='query-display'>
                    {queries.map((message, i) => <li key={i}>{message.text}, {message.rating.toString()}</li>)}
                </ul>
            {/* </span> */}
            
        </div>
        </div>
    )
}