import './adminPanel.css';
import {FormEvent, Key, useState} from "react";

type Message = {
    text: String;
    rating: "Positive" | "Negative" | "Not Rated";
}

export default function ReviewQueries() {

    const [queries, setQueries] = useState<Message[]>([]);

    const convertNumericalRating = (r: -1 | 0 | 1) => {
        if (r == -1) {
            return "Negative"
        } else if (r == 0) {
            return "Not Rated"
        } else {
            return "Positive"
        }
    }

    const fetchMessage = async(r: Number) => {
        try {
            const msgs: Message[] = [];

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


            const res = await response.json();
            for (let i in res) {
                const m: Message = {
                    text: res[i].text,
                    rating: convertNumericalRating(res[i].rating)
                }
                msgs.push(m)
            }

            return msgs;
        } catch (error) {
            const msgs: Message[] = [];
            console.error("Error: ", error);
            return msgs;
        }
    }

    const fetchAll = async() => {
        try {
            const msgs: Message[] = []

            const response = await fetch(`https://onc-assistant-822f952329ee.herokuapp.com/api/messages-all`,
                {
                method: "GET",     
                headers: {
                    "Content-Type": "application/json",
                }  
            })

            if (!response.ok) {
                throw new Error("API request failed");
            }

            const res = await response.json();
            
            for (let i in res) {
                const m: Message = {
                    text: res[i].text,
                    rating: convertNumericalRating(res[i].rating)
                }
                msgs.push(m)
            }

            return msgs;
        } catch (error) {
            const msgs: Message[] = []
            console.error("Error: ", error);
            return msgs;
        }
    }

    const retrieveQueries = async(e: FormEvent) => {
        const event = e.target as HTMLFormElement;
        const rate = event.value;
    
        let retrieved: Message[] = []
        
        if (rate == 2) {
            retrieved =await(fetchAll())
            
        } else {
            retrieved =await(fetchMessage(rate))
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
                <ul className='query-display'>
                    {queries.map((message, i) => <li key={i}>{message.rating}: {message.text}</li>)}
                </ul>            
        </div>
        </div>
    )
}