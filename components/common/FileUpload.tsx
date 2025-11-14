import React, { useRef, useState } from 'react';
import { ArrowUpTrayIcon, PhotoIcon, VideoCameraIcon, DocumentIcon } from '@heroicons/react/24/solid';

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    accept: string;
    fileType?: 'image' | 'video' | 'any';
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, accept, fileType = 'any' }) => {
    const [dragging, setDragging] = useState(false);
    const [fileName, setFileName] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setFileName(file.name);
            onFileSelect(file);
        }
    };

    const handleDragEvents = (e: React.DragEvent<HTMLLabelElement>, isEntering: boolean) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(isEntering);
    };

    const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            setFileName(file.name);
            onFileSelect(file);
        }
    };

    const getIcon = () => {
        switch(fileType) {
            case 'image': return <PhotoIcon className="h-12 w-12 text-gray-500" />;
            case 'video': return <VideoCameraIcon className="h-12 w-12 text-gray-500" />;
            default: return <DocumentIcon className="h-12 w-12 text-gray-500" />;
        }
    }

    return (
        <label
            htmlFor="file-upload"
            className={`flex flex-col items-center justify-center w-full h-48 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-800 hover:bg-gray-700 transition-colors ${dragging ? 'border-indigo-500 bg-gray-700' : ''}`}
            onDragEnter={(e) => handleDragEvents(e, true)}
            onDragLeave={(e) => handleDragEvents(e, false)}
            onDragOver={(e) => handleDragEvents(e, true)}
            onDrop={handleDrop}
        >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                {fileName ? (
                    <>
                        {getIcon()}
                        <p className="mt-2 text-sm text-gray-400"><span className="font-semibold">{fileName}</span></p>
                        <p className="text-xs text-gray-500">Click or drag to replace</p>
                    </>
                ) : (
                    <>
                        <ArrowUpTrayIcon className="h-10 w-10 text-gray-500 mb-3" />
                        <p className="mb-2 text-sm text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                        <p className="text-xs text-gray-500">{accept.replace(/,/g, ', ')}</p>
                    </>
                )}
            </div>
            <input
                id="file-upload"
                type="file"
                className="hidden"
                accept={accept}
                onChange={handleFileChange}
                ref={inputRef}
            />
        </label>
    );
};