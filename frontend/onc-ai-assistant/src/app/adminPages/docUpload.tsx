'use client';
import "./adminPanel.css";
import { useRef, useState } from "react";


export default function DocUpload() {
    const browseInput = useRef<HTMLInputElement>(null);
    const [files, setFiles] = useState<File[]>([]);
    const [isDragging, setDragging] = useState(false);
    const [hasFiles, setHasFiles] = useState(false);

    const addFiles = (filesToAdd: File[]) => {
        const tempFiles = files;
        for (const i in filesToAdd) {
            tempFiles.push(filesToAdd[i])
        }
        setHasFiles(true);
        setFiles(tempFiles);
    }

    const handleDrop = (e: any) => {
        e.preventDefault();
        addFiles(Array.from(e.dataTransfer.files));
        setDragging(false);
    }

    const handleDragOver = (e:any) => {
        e.preventDefault();
        setDragging(true)
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
    }

    
    const uploadFiles = () => {
        for (let i = 0; i < files.length; i++) {
            let file: File = files[i];
            console.log(file.name)
        }
        setFiles([]); //clear files
        setHasFiles(false);
    }

    return (
        <div className="module">
            <h2>Upload Documents</h2>
            <div className="fileDrop" 
                onDragOver={handleDragOver} 
                onDrop={handleDrop}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragExit}
                style={{backgroundColor: isDragging ? "#d3d3d3" : "white",  borderStyle: (hasFiles ? "solid" : "dashed")}}
            > 
                <div className="fileList" style={{overflow: (hasFiles ? "auto" : "none")}}>
                    <p style={{display: (hasFiles ? "none" : "block"), padding: "5rem 0", color:"#7F7F7F"}}><i>
                    Drag and drop/upload files here to enhance the assistant's model.
                    </i></p>
                    {Array.from(files).map((file, i) => <p key={i} style={{color: "black"}}>{file.name}</p>)}
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

