import React, { useState, useCallback, FC } from 'react';
import { generateImage, editImage } from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { FileUpload } from '../common/FileUpload';
import { Spinner } from '../common/Spinner';
import { PhotoIcon } from '@heroicons/react/24/outline';

type ToolMode = 'generate' | 'edit';
const aspectRatios = ["1:1", "16:9", "9:16", "4:3", "3:4"];

export const ImageTools: FC = () => {
    const [mode, setMode] = useState<ToolMode>('generate');
    const [prompt, setPrompt] = useState('');
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [sourceImage, setSourceImage] = useState<{ file: File; base64: string; mimeType: string } | null>(null);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileSelect = useCallback(async (file: File) => {
        try {
            const base64 = await fileToBase64(file);
            setSourceImage({ file, base64, mimeType: file.type });
            setGeneratedImage(null); // Clear previous result when new image is uploaded
            setError(null);
        } catch (err) {
            setError('Failed to read the selected file.');
        }
    }, []);

    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt || isLoading || (mode === 'edit' && !sourceImage)) return;
        
        setIsLoading(true);
        setError(null);
        setGeneratedImage(null);

        try {
            let resultImage: string;
            if (mode === 'generate') {
                resultImage = await generateImage(prompt, aspectRatio);
            } else if (sourceImage) {
                resultImage = await editImage(prompt, sourceImage.base64, sourceImage.mimeType);
            } else {
                throw new Error("Source image is required for editing.");
            }
            setGeneratedImage(resultImage);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [prompt, isLoading, mode, sourceImage, aspectRatio]);
    
    const renderGeneratorForm = () => (
        <div className="space-y-4">
            <div>
                <label htmlFor="aspectRatio" className="block text-sm font-medium text-gray-300 mb-1">Aspect Ratio</label>
                <select
                    id="aspectRatio"
                    value={aspectRatio}
                    onChange={(e) => setAspectRatio(e.target.value)}
                    className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    disabled={isLoading}
                >
                    {aspectRatios.map(ar => <option key={ar} value={ar}>{ar}</option>)}
                </select>
            </div>
        </div>
    );
    
    const renderEditorForm = () => (
        <div className="space-y-4">
            <FileUpload onFileSelect={handleFileSelect} accept="image/png, image/jpeg" fileType="image"/>
             {sourceImage && (
                <div className="mt-4">
                    <img src={URL.createObjectURL(sourceImage.file)} alt="Selected for editing" className="max-h-48 rounded-lg mx-auto" />
                </div>
            )}
        </div>
    );

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Image Studio</h2>
            <p className="text-gray-400">Generate visualizations of model architectures or edit existing diagrams with simple text prompts.</p>
            
            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg p-1 w-min">
                <button onClick={() => setMode('generate')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'generate' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Generate</button>
                <button onClick={() => setMode('edit')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'edit' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Edit</button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <h3 className="text-xl font-semibold">{mode === 'generate' ? 'Generate New Image' : 'Edit Existing Image'}</h3>
                        <div>
                            <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
                            <textarea
                                id="prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder={mode === 'generate' ? "e.g., A diagram of a convolutional neural network" : "e.g., Add a retro filter"}
                                className="w-full h-24 p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500"
                                disabled={isLoading}
                            />
                        </div>
                        {mode === 'generate' ? renderGeneratorForm() : renderEditorForm()}
                        <Button type="submit" isLoading={isLoading} disabled={!prompt || (mode === 'edit' && !sourceImage)} className="w-full">
                           {mode === 'generate' ? 'Generate Image' : 'Apply Edit'}
                        </Button>
                    </form>
                </Card>
                <Card className="flex items-center justify-center min-h-[300px]">
                    {isLoading && <Spinner text="Processing image..." />}
                    {error && <p className="text-red-400 text-center">Error: {error}</p>}
                    {generatedImage && !isLoading && (
                        <div className="space-y-4 text-center">
                            <h3 className="text-xl font-semibold">Result</h3>
                            <img src={generatedImage} alt="Generated result" className="max-w-full max-h-96 rounded-lg shadow-lg" />
                        </div>
                    )}
                     {!isLoading && !generatedImage && !error && (
                        <div className="text-center text-gray-500">
                             <PhotoIcon className="h-16 w-16 mx-auto mb-2"/>
                             Your generated or edited image will appear here.
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
};