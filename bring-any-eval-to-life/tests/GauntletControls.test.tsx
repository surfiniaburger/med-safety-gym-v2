import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import React from 'react';
import { GauntletView } from '../components/Gauntlet/GauntletView';

// Mock Three.js/Fiber components to avoid Canvas issues in JSDOM
vi.mock('@react-three/fiber', () => ({
    Canvas: () => <div data-testid="mock-canvas" />,
    useFrame: vi.fn(),
}));

describe('Gauntlet Camera Controls', () => {
    const defaultProps = {
        rewards: [10, 20, 30],
        activeStepIndex: 0,
        solvedNodes: [],
        onIntervene: vi.fn(),
        onActiveStepChange: vi.fn(),
        onClose: vi.fn(),
    };

    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('renders navigation controls after warmup', () => {
        render(<GauntletView {...defaultProps} />);
        
        // Advance past warmup (6.5s)
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        // Open controls dropdown
        const controlsButton = screen.getByText(/Controls/i);
        fireEvent.click(controlsButton);

        expect(screen.getByText(/Free Camera/i)).toBeDefined();
        expect(screen.getByLabelText(/Neural Intensity/i)).toBeDefined();
        expect(screen.getByLabelText(/Sim Speed/i)).toBeDefined();
    });

    it('activates manual mode when the Free Camera button is clicked', () => {
        render(<GauntletView {...defaultProps} />);
        
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        // Open controls dropdown
        const controlsButton = screen.getByText(/Controls/i);
        fireEvent.click(controlsButton);

        const freeCamButton = screen.getByText(/Free Camera/i);
        fireEvent.click(freeCamButton);

        // Should show "Manual Mode" indicator and "Reset Camera" button
        // Note: Reset Camera is outside the dropdown in the original code, 
        // but let's check where it is now.
        expect(screen.getByText(/Manual Mode/i)).toBeDefined();
    });
});
