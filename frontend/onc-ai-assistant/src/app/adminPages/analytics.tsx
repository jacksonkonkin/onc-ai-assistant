import './adminPanel.css';
import {BarChart, Bar, CartesianGrid, XAxis, YAxis} from 'recharts';
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

type RatingFrequency = {
    rating: String,
    count: Number
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

        // const posData: AnalyticsData = {key: "Positive", val: posCount}
        // const neutralData: AnalyticsData = {key: "Not Rated", val: neutralCount}
        // const negData: AnalyticsData = {key: "Negative", val: negCount}

        // const frequency: AnalyticsDisplay = {metric: "Rating Frequency", data: [posData, neutralData, negData]}
        // return frequency;

        // const frequency: RatingFrequency = {Positive: posCount, Unrated: neutralCount, Negative: negCount}

        const posFreq: RatingFrequency = {rating: "Positive", count: posCount}
        const neutralFreq: RatingFrequency = {rating: "Not Rated", count: neutralCount}
        const negFreq: RatingFrequency = {rating: "Negative", count: negCount}

        return [posFreq, neutralFreq, negFreq]
        
    }

    

    
    return(
        <div className="module">
        <h2>View Analytics</h2>
        <button onClick={setupData}>Reload Data</button>
        <div className="analytics"> 
            <div className="chartDisplay"></div>

            <BarChart width={730} height={250} data={ratingFrequency()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="rating" />
                <YAxis />
                <Bar dataKey="count" fill="#123253"/>
            </BarChart>

            </div>
        </div>
    );
}