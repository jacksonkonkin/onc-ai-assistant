import './adminPanel.css';
import {FormEvent, Key, useState} from "react";
import data from './messages.json';

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

    const [queries, setQueries] = useState<Message[]>(sampleMessages);

    // temp function to test handling jsons
    const rateFilter = (r: Number) => {
        const json: any = data
        console.log(json)
        const messages: Message[] = []

        for (let i in json) {
            if (json[i].rating == r) {
                const m: Message = {text: json[i].text, rating: json[i].rating}
                messages.push(m)
            }
        }
        return messages

    }

    const fetchMessage = async(r: Number) => {
        try {
            const response = await fetch(`https://onc-assistant-822f952329ee.herokuapp.com/api/messages-by-rating?rating=${r}`,
                {
                method: "GET",     
                headers: {
                    "Content-Type": "application/json",
                }  
            })

            if (!response.ok) {
                throw new Error("API request failed");
            }

            const json = await response.json();
            // const messages = JSON.parse(json);
            // console.log(messages)
            // need to format for display still
            // return messages.response;
        } catch (error) {
            console.error("Error: ", error);
            return "Error retrieving messages.";
        }
    }

    //eventually this will call the messages API endpoint
    const retrieveQueries = (e: FormEvent) => {
        const event = e.target as HTMLFormElement;
        const rate = event.value;
    
        let retrieved: Message[] = []
        
        if (rate == 2) {
            // fetch all
            const pos = rateFilter(1)
            const neutral = rateFilter(0)
            const neg = rateFilter(-1)

            retrieved = pos.concat(neutral, neg) 

        } else {
        //    fetchMessage(rate); //fetch only for the rating chosen
            retrieved = rateFilter(rate)
        }
        
        setQueries(retrieved);
    }

    
    return(
        <div className="module">
        <h2>Review User Feedback & Frequent Queries</h2>
        <div className="frequent-queries">
            <div className="rating-filter">
                <label htmlFor="rating">Select messages to show:</label>
                <select id="rating" name="rating" onChange={(e) => retrieveQueries(e)} defaultValue={2}>
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