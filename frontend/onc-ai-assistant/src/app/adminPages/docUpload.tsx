'use client';
import "./adminPanel.css";
import { useRef, useState } from "react";


export default function DocUpload() {
    const browseInput = useRef<HTMLInputElement>(null);
    const [files, setFiles] = useState<File[]>([]);
    const [isDragging, setDragging] = useState(false);

    const addFiles = (filesToAdd: File[]) => {
        const tempFiles = files;
        for (const i in filesToAdd) {
            tempFiles.push(filesToAdd[i])
        }
        setFiles(tempFiles);
    }

    const handleDrop = (e: any) => {
        e.preventDefault();
        addFiles(Array.from(e.dataTransfer.files));
        setDragging(false);
    }

    const handleDragOver = (e:any) => {
        e.preventDefault();
    }

    const handleDragEnter = () => setDragging(true);
    const handleDragExit = () => setDragging(false);

    const triggerBrowse = (e:any) => {
        e.preventDefault();
        if (browseInput.current) {
            browseInput.current.click();
        }
    }

    const addFromBrowse = (e: any) => {
        addFiles(Array.from(e.target.files))
        console.log("add from browse")
    }

    // eventually this will contain sending to backend (or wherever they need to go)
    const uploadFiles = () => {
        console.log("uploading: " + files.length);
        for (let i = 0; i < files.length; i++) {
            let file: File = files[i];
            console.log(file.name)
        }
        setFiles([]);
    }

    //overflow of filenames listed in the drag and drop isn't being handled but it can take 9 or 10 filenames and display them currently
    return (
        <div className="module">
            <h2>Upload Documents</h2>
            <div className="fileDrop" 
                onDragOver={handleDragOver} 
                onDrop={handleDrop}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragExit}
                style={{backgroundColor: isDragging ? "#d3d3d3" : "white" }}
            > 
                <p style={{visibility: (files.length == 0 ? "visible" : "hidden")}}><i>
                    Drag and drop/upload files here to enhance the assistant's model.
                </i></p>
                <div style={{visibility: (files.length == 0 ? "hidden" : "visible")}}>
                    {Array.from(files).map(file => <p>{file.name}</p>)}
                </div>

            </div>
            <div className="upload"> 
                <input ref={browseInput} type="file" onChange={addFromBrowse} hidden multiple/>
                <button onClick={triggerBrowse}>Browse</button>
                <button onClick={uploadFiles}>Upload</button>
            </div>
        </div>
    );
}

