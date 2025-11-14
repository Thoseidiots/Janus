import React from 'react';

interface SpinnerProps {
    size?: 'sm' | 'md' | 'lg';
    text?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({ size = 'md', text }) => {
    const sizeClasses = {
        sm: 'h-6 w-6',
        md: 'h-8 w-8',
        lg: 'h-12 w-12',
    };
    
    return (
        <div className="flex flex-col items-center justify-center gap-4">
            <div className={`animate-spin rounded-full ${sizeClasses[size]} border-b-2 border-t-2 border-indigo-500`}></div>
            {text && <p className="text-gray-400">{text}</p>}
        </div>
    );
};