import './adminPanel.css';
import {BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip} from 'recharts';
import {useState} from 'react';

type Message = {
    text: String;
    rating: -1 | 0 | 1;
    timestamp: Date;
}

type RatingFrequency = {
    rating: String,
    Count: Number
}

export default function Analytics() {
    const [msgs, setMsgs] = useState<Message[]>([]);

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
                    rating: res[i].rating,
                    timestamp: res[i].timestamp
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

    const setupData = async() => {
        setMsgs(await fetchAll());
    }

    const ratingFrequency = () => {
        let posCount = 0, neutralCount = 0, negCount = 0;
        for (let i in msgs) {
            if (msgs[i].rating == 1) {
                posCount++;
            } else if (msgs[i].rating == 0) {
                neutralCount++;
            } else if (msgs[i].rating == -1) {
                negCount++;
            }
        }
        const posFreq: RatingFrequency = {rating: "Positive", Count: posCount}
        const neutralFreq: RatingFrequency = {rating: "Not Rated", Count: neutralCount}
        const negFreq: RatingFrequency = {rating: "Negative", Count: negCount}
        return [posFreq, neutralFreq, negFreq]
    }
    
    return(
        <div className="module">
        <h2>View Analytics</h2>
        <div className="analytics"> 
            <div className="ratingFrequency">
                <h4> Rating Frequency </h4>
                <BarChart width={500} height={500} data={ratingFrequency()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="rating" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="Count" fill="#123253"/>
                </BarChart>
            </div>
            <button className="reloadData" onClick={setupData}>Reload Data</button>
        </div>
        </div>
    );
}