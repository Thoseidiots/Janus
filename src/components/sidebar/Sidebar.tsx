import React, { FC, ElementType } from 'react';
import { CubeTransparentIcon } from '@heroicons/react/24/outline';

interface Tool {
    name: string;
    icon: ElementType;
}

interface SidebarProps {
    tools: Tool[];
    activeToolName: string;
    onToolSelect: (name: string) => void;
}

export const Sidebar: FC<SidebarProps> = ({ tools, activeToolName, onToolSelect }) => {
    return (
        <aside className="w-64 bg-gray-900/70 backdrop-blur-sm border-r border-gray-800 flex flex-col flex-shrink-0">
            <div className="h-16 flex items-center justify-center border-b border-gray-800 flex-shrink-0">
                <CubeTransparentIcon className="h-8 w-8 text-indigo-400" />
                <h1 className="text-xl font-bold ml-2">ThinkingBrain</h1>
            </div>
            <nav className="flex-grow overflow-y-auto">
                <ul className="p-2 space-y-1">
                    {tools.map((tool) => (
                        <li key={tool.name}>
                            <button
                                onClick={() => onToolSelect(tool.name)}
                                className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors ${
                                    activeToolName === tool.name
                                        ? 'bg-indigo-600 text-white font-semibold'
                                        : 'text-gray-300 hover:bg-gray-800'
                                }`}
                            >
                                <tool.icon className="h-5 w-5" />
                                <span>{tool.name}</span>
                            </button>
                        </li>
                    ))}
                </ul>
            </nav>
        </aside>
    );
};