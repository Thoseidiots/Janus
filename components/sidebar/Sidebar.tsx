import React, { useState } from 'react';
import { Feature, FeatureID } from '../../App';

interface SidebarProps {
    features: Feature[];
    activeFeature: FeatureID;
    setActiveFeature: (feature: FeatureID) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ features, activeFeature, setActiveFeature }) => {
    const [isOpen, setIsOpen] = useState(false);

    const handleItemClick = (featureId: FeatureID) => {
        setActiveFeature(featureId);
        setIsOpen(false);
    };

    const sidebarContent = (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '16px', borderBottom: '1px solid #1e293b'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 22 }}>🧠</span>
                    <span style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', fontFamily: "'Syne',sans-serif" }}>
                        Janus Studio
                    </span>
                </div>
                <button
                    onClick={() => setIsOpen(false)}
                    style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 18, padding: 4 }}
                >
                    ✕
                </button>
            </div>
            <nav style={{ flex: 1, padding: '12px 8px', overflowY: 'auto' }}>
                {features.map((feature) => {
                    const isActive = activeFeature === feature.id;
                    return (
                        <button
                            key={feature.id}
                            onClick={() => handleItemClick(feature.id)}
                            title={feature.description}
                            style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 10,
                                padding: '8px 12px',
                                marginBottom: 4,
                                borderRadius: 8,
                                border: 'none',
                                cursor: 'pointer',
                                textAlign: 'left',
                                background: isActive ? '#6366f1' : 'transparent',
                                color: isActive ? '#ffffff' : '#94a3b8',
                                fontFamily: "'Space Mono','Courier New',monospace",
                                fontSize: 11,
                                fontWeight: isActive ? 700 : 400,
                                transition: 'background 0.15s, color 0.15s',
                            }}
                            onMouseEnter={e => {
                                if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = '#1e293b';
                            }}
                            onMouseLeave={e => {
                                if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                            }}
                        >
                            <span style={{ fontSize: 16, flexShrink: 0 }}>{feature.icon}</span>
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {feature.label}
                            </span>
                        </button>
                    );
                })}
            </nav>
        </div>
    );

    return (
        <>
            {/* Mobile toggle button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    display: 'none',
                    position: 'fixed', top: 12, left: 12, zIndex: 30,
                    padding: '6px 10px', background: '#0a1222',
                    border: '1px solid #1e293b', borderRadius: 6,
                    color: '#e2e8f0', cursor: 'pointer', fontSize: 18,
                }}
                className="mobile-menu-btn"
            >
                ☰
            </button>

            {/* Mobile overlay */}
            {isOpen && (
                <div
                    onClick={() => setIsOpen(false)}
                    style={{
                        position: 'fixed', inset: 0, zIndex: 20,
                        background: 'rgba(6,13,23,0.75)',
                    }}
                />
            )}

            {/* Mobile sidebar */}
            <aside style={{
                position: 'fixed', top: 0, left: 0, height: '100%', width: 220,
                background: '#0a1222', borderRight: '1px solid #1e293b',
                transform: isOpen ? 'translateX(0)' : 'translateX(-100%)',
                transition: 'transform 0.2s ease',
                zIndex: 25,
            }}
                className="mobile-sidebar"
            >
                {sidebarContent}
            </aside>

            {/* Desktop sidebar */}
            <aside style={{
                width: 220, flexShrink: 0,
                background: '#0a1222', borderRight: '1px solid #1e293b',
                height: '100vh', position: 'sticky', top: 0,
                overflowY: 'auto',
            }}
                className="desktop-sidebar"
            >
                {sidebarContent}
            </aside>

            <style>{`
                @media (max-width: 1023px) {
                    .desktop-sidebar { display: none !important; }
                    .mobile-menu-btn { display: block !important; }
                }
                @media (min-width: 1024px) {
                    .mobile-sidebar { display: none !important; }
                }
            `}</style>
        </>
    );
};
