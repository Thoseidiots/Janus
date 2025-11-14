import React from 'react';

interface ProgressBarProps {
    value: number;
    color: string;
    label: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ value, color, label }) => (
    <div>
        <div className="flex justify-between mb-1">
            <span className="text-sm font-medium text-gray-300">{label}</span>
            <span className="text-sm font-medium text-gray-400">{`${Math.round(value * 100)}%`}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div 
                className={`${color} h-2.5 rounded-full transition-all duration-500`} 
                style={{ width: `${value * 100}%` }}
            />
        </div>
    </div>
);