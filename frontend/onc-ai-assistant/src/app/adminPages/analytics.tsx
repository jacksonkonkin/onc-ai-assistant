import './adminPanel.css';
// import {BarChart, Bar, CartesianGrid} from 'recharts';
import data from './messages.json';
import {Component, useState} from 'react';

type Message = {
    text: String;
    rating: Number;
    timestamp: Date;
}

type AnalyticsData = {
    key: String,
    val: any
}

type AnalyticsDisplay = {
    metric: String;
    data: AnalyticsData[]
}

export default function Analytics() {
    const [msgs, setMsgs] = useState<Message[]>([]);

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
            // will finish this when json parsing is working
        } catch (error) {
            console.error("Error: ", error);
            return "Error retrieving messages.";
        }
    }

    const setupData = () => {
        //fetch message calls for all messages
        const json: any = data
        const messages: Message[] =[]
        for (let i in json) {
            const m: Message = {
                text: json[i].text,
                rating: json[i].rating,
                timestamp: json[i].timestamp
            }
            messages.push(m)
        }
        setMsgs(messages);
    }

    const ratingFrequency = () => {
        const pos = 1, neutral = 0, neg = -1;
        let posCount = 0, neutralCount = 0, negCount = 0;
        for (let i in msgs) {
            if (msgs[i].rating == pos) {
                posCount++;
            } else if (msgs[i].rating == neutral) {
                neutralCount++;
            } else if (msgs[i].rating == neg) {
                negCount++;
            }
        }

        const posData: AnalyticsData = {key: "Positive", val: posCount}
        const neutralData: AnalyticsData = {key: "Not Rated", val: neutralCount}
        const negData: AnalyticsData = {key: "Negative", val: negCount}

        const frequency: AnalyticsDisplay = {metric: "Rating Frequency", data: [posData, neutralData, negData]}
        return frequency;
        
    }

    

    
    return(
        <div className="module">
        <h2>View Analytics</h2>
        <div className="analytics"> 
            <button onClick={setupData}>Reload Data</button>
            <div className="chartDisplay"></div>
            <h2>{ratingFrequency().metric}</h2>
            {ratingFrequency().data.map((datapoint, i) => <li key={i}>{datapoint.key}: {datapoint.val}</li>)}
        </div>
        </div>
    );
}