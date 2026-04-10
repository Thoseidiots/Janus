
import React, { useState } from 'react';
import { generateText } from '../../services/geminiService';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import ReactMarkdown from 'react-markdown';
import { ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/solid';
import { InformationCircleIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

type GenerationMode = 'fast' | 'standard' | 'deep';
type TuningStrategy = 'defaults' | 'grid' | 'random' | 'bayesian' | 'evolutionary' | 'hyperband';


const modeConfig = {
    fast: { model: 'gemini-2.5-flash-lite', title: 'Fast Mode', useThinking: false },
    standard: { model: 'gemini-2.5-flash', title: 'Standard Mode', useThinking: false },
    deep: { model: 'gemini-2.5-pro', title: 'Deep Dive', useThinking: true },
} as const;


export const HyperparameterGenerator: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [mode, setMode] = useState<GenerationMode>('standard');
    const [strategy, setStrategy] = useState<TuningStrategy>('defaults');
    const [result, setResult] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showExplanation, setShowExplanation] = useState(false);

    const [generatedCode, setGeneratedCode] = useState('');
    const [isCodeLoading, setIsCodeLoading] = useState(false);
    const [isCopied, setIsCopied] = useState(false);

    const handleReset = () => {
        setResult('');
        setGeneratedCode('');
        setError(null);
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt || isLoading) return;

        setIsLoading(true);
        handleReset();

        try {
            const config = modeConfig[mode];
            let fullPrompt = '';
            
            switch(strategy) {
                case 'grid':
                    fullPrompt = `As an expert machine learning engineer, define a hyperparameter search space for a Grid Search strategy for the following scenario. Provide the output in a clean markdown table with columns for Hyperparameter, Search Values (e.g., [16, 32, 64]), and Rationale. Scenario: ${prompt}`;
                    break;
                case 'random':
                    fullPrompt = `As an expert machine learning engineer, define a hyperparameter search space for a Random Search strategy for the following scenario. Provide the output in a clean markdown table with columns for Hyperparameter, Distribution/Range (e.g., Uniform(0.0001, 0.01)), and Rationale. Scenario: ${prompt}`;
                    break;
                case 'bayesian':
                    fullPrompt = `As an expert machine learning engineer, define a hyperparameter search space for a Bayesian Optimization strategy for the following scenario. Provide the output in a clean markdown table with columns for Hyperparameter, Search Range/Bounds (e.g., [1e-5, 1e-1]), and Rationale. Scenario: ${prompt}`;
                    break;
                case 'evolutionary':
                    fullPrompt = `As an expert machine learning engineer, define a hyperparameter search space for an Evolutionary Algorithm strategy for the following scenario. Also, suggest key evolutionary parameters like population size, mutation rate, and crossover rate. Provide the output in a clean markdown format with tables. Scenario: ${prompt}`;
                    break;
                case 'hyperband':
                    fullPrompt = `As an expert machine learning engineer, explain how to set up a ASHA/HyperBand strategy for the following scenario. Define the hyperparameter search space and suggest values for key parameters like the reduction factor (eta) and minimum/maximum resources per trial. Provide the output in a clean markdown format with tables. Scenario: ${prompt}`;
                    break;
                case 'defaults':
                default:
                    fullPrompt = `As an expert machine learning engineer, generate a set of recommended hyperparameters for the following scenario. Provide the output in a clean markdown table with columns for Hyperparameter, Recommended Value, and a brief Rationale. Scenario: ${prompt}`;
                    break;
            }
            
            const response = await generateText(fullPrompt, config.model, config.useThinking);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleGenerateCode = async () => {
        if (!prompt || !result || isCodeLoading) return;
        setIsCodeLoading(true);
        setGeneratedCode('');
        setError(null);

        try {
            const codePrompt = `
                As an expert machine learning engineer, write a Python code snippet to implement a hyperparameter search for the following scenario.
                - Use a relevant, popular library (e.g., Scikit-learn for Grid/Random Search, Optuna for Bayesian, DEAP for Evolutionary).
                - The code should be self-contained and easy to understand, with placeholders for the model and data.
                - It must use the provided search space.

                **Scenario:** ${prompt}

                **Tuning Strategy:** ${strategy}

                **Generated Search Space:**
                ${result}

                Provide only the Python code in a single markdown block.
            `;
            const response = await generateText(codePrompt, 'gemini-2.5-pro', false);
            // Clean up markdown code block delimiters
            const cleanedResponse = response.replace(/^```python\n/, '').replace(/\n```$/, '');
            setGeneratedCode(cleanedResponse);
        } catch (err) {
            setError(err instanceof Error ? `Failed to generate code: ${err.message}`: 'An unknown error occurred while generating code.');
        } finally {
            setIsCodeLoading(false);
        }
    };

    const handleCopy = () => {
        if (generatedCode) {
            navigator.clipboard.writeText(generatedCode);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        }
    };
    
    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Hyperparameter Generator</h2>
            <p className="text-gray-400">Describe your machine learning model or problem, and let Gemini suggest optimal hyperparameters. Choose a mode based on your need for speed vs. depth of analysis, and select a tuning strategy for more advanced recommendations.</p>
            
            <div className="mt-2">
                <button 
                    onClick={() => setShowExplanation(!showExplanation)}
                    className="flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                    <InformationCircleIcon className="h-5 w-5" />
                    <span>What's the difference between hyperparameters and parameters?</span>
                    {showExplanation ? <ChevronUpIcon className="h-4 w-4" /> : <ChevronDownIcon className="h-4 w-4" />}
                </button>
                {showExplanation && (
                    <Card className="mt-2 bg-gray-900/50">
                        <div className="prose prose-sm prose-invert max-w-none text-gray-300">
                            <h4 className="text-gray-100">The Developer's Role: Choosing the Hyperparameters</h4>
                            <p>
                                An AI developer decides the architecture (the blueprint) of the neural network. These choices are called <strong>hyperparameters</strong> (external configuration variables) and they directly determine the maximum possible number of parameters the AI can generate.
                            </p>
                            <h4 className="text-gray-100">The AI's Role: Making the Parameters (Weights and Biases)</h4>
                            <p>
                                The <strong>parameters</strong> themselves are the vast set of numerical values (called weights and biases) that define the model's knowledge. The AI creates and refines these parameters in three main steps during the training process:
                            </p>
                            <ol>
                                <li>
                                    <strong>Initialization (Creation):</strong> The model begins by filling all the weights and biases with random numerical values. This is the initial "creation" of the parameters. The model starts with zero knowledge.
                                </li>
                                <li>
                                    <strong>Forward Pass and Error Calculation:</strong> The model processes the training data (e.g., millions of sentences or images). It makes a prediction and compares it to the correct answer. The difference is the error, which is calculated by the loss function.
                                </li>
                                <li>
                                    <strong>Backpropagation and Adjustment (Refinement):</strong> Using a mathematical process called Backpropagation, the error is sent backward through the network. Algorithms like Gradient Descent precisely calculate how much each individual weight and bias (parameter) needs to be adjusted to reduce that error. The model makes tiny, iterative adjustments to these numbers.
                                </li>
                            </ol>
                            <p>
                                By repeating this process trillions of times, the AI system continuously tunes its own parameters from random numbers into the highly specific values that encode its ability to understand language and generate images. The AI literally learns the values of its parameters.
                            </p>
                        </div>
                    </Card>
                )}
            </div>

            <Card>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">Scenario Description</label>
                        <textarea
                            id="prompt"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="e.g., I'm training a ResNet50 for image classification on CIFAR-10..."
                            className="w-full h-32 p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            disabled={isLoading}
                        />
                    </div>
                     <div>
                        <label htmlFor="strategy" className="block text-sm font-medium text-gray-300 mb-1">Tuning Strategy</label>
                        <select
                            id="strategy"
                            value={strategy}
                            onChange={(e) => setStrategy(e.target.value as TuningStrategy)}
                            className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            disabled={isLoading}
                        >
                            <option value="defaults">Single Value (Defaults)</option>
                            <option value="grid">Grid Search</option>
                            <option value="random">Random Search</option>
                            <option value="bayesian">Bayesian Optimization</option>
                            <option value="evolutionary">Evolutionary Algorithms</option>
                            <option value="hyperband">ASHA / HyperBand</option>
                        </select>
                    </div>
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                         <div className="flex items-center space-x-2 bg-gray-700 rounded-lg p-1">
                            {(Object.keys(modeConfig) as GenerationMode[]).map((m) => (
                                <button
                                    key={m}
                                    type="button"
                                    onClick={() => setMode(m)}
                                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${mode === m ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-600'}`}
                                >
                                    {modeConfig[m].title}
                                </button>
                            ))}
                        </div>
                        <Button type="submit" isLoading={isLoading} disabled={!prompt}>
                            Generate Parameters
                        </Button>
                    </div>
                </form>
            </Card>

            {isLoading && (
                <div className="flex justify-center py-8">
                    <Spinner text="Generating suggestions..." />
                </div>
            )}

            {error && (
                <Card>
                    <p className="text-red-400">Error: {error}</p>
                </Card>
            )}

            {result && !isLoading && (
                <Card>
                    <h3 className="text-xl font-semibold mb-4">{strategy === 'defaults' ? 'Generated Hyperparameters' : 'Hyperparameter Search Space'}</h3>
                    <div className="prose prose-invert max-w-none prose-table:w-full prose-td:py-2 prose-td:px-4 prose-th:px-4">
                      <ReactMarkdown>{result}</ReactMarkdown>
                    </div>
                    <div className="mt-6 text-right">
                        <Button onClick={handleGenerateCode} isLoading={isCodeLoading}>
                            Export Code
                        </Button>
                    </div>
                </Card>
            )}

            {isCodeLoading && (
                 <div className="flex justify-center py-8">
                    <Spinner text="Generating code snippet..." />
                </div>
            )}

            {generatedCode && !isCodeLoading && (
                 <Card>
                    <div className="flex justify-between items-center mb-2">
                        <h3 className="text-xl font-semibold">Generated Python Code</h3>
                        <button 
                            onClick={handleCopy}
                            className="flex items-center gap-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors"
                        >
                            {isCopied ? <CheckIcon className="h-4 w-4 text-green-400" /> : <ClipboardDocumentIcon className="h-4 w-4" />}
                            {isCopied ? 'Copied!' : 'Copy'}
                        </button>
                    </div>
                    <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
                        <code>
                            {generatedCode}
                        </code>
                    </pre>
                </Card>
            )}
        </div>
    );
};
