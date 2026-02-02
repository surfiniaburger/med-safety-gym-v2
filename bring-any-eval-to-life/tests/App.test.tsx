import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
import * as githubService from '../services/github';
import * as geminiService from '../services/gemini';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the services
vi.mock('../services/github');
vi.mock('../services/gemini');
vi.mock('../components/Gauntlet/GauntletView', () => ({
    GauntletView: vi.fn(({ onIntervene }) => (
        <div data-testid="gauntlet-view">
            Gauntlet View Mock
            <button onClick={() => onIntervene(0)}>Intervene</button>
        </div>
    ))
}));

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
        {children}
    </QueryClientProvider>
);

describe('App Integration', () => {
    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('renders artifacts and triggers simulation generation on selection', async () => {
        const mockArtifacts: githubService.EvaluationArtifact[] = [
            {
                id: '1',
                name: 'test-artifact.json',
                path: 'results/test-artifact.json',
                sha: 'sha1',
                url: '', html_url: '', download_url: '',
                content: {
                    safety_score: 0.85,
                    status: 'SAFE',
                    summary: 'Simulation test'
                }
            }
        ];

        vi.mocked(githubService.fetchEvaluationArtifacts).mockResolvedValue(mockArtifacts);
        vi.mocked(geminiService.bringToLife).mockResolvedValue('<html><body>Clinical Simulation</body></html>');

        render(<App />, { wrapper });

        // 1. Verify artifacts are loaded and displayed
        await waitFor(() => {
            expect(screen.getByText(/test artifact/i)).toBeDefined();
        });

        // 2. Select an artifact
        const card = screen.getByText(/test artifact/i).closest('div');
        if (card) fireEvent.click(card);

        // 3. Verify Gauntlet View is mounted but Gemini bridge is NOT called yet (On-Demand)
        await waitFor(() => {
            expect(screen.getByTestId('gauntlet-view')).toBeDefined();
        });
        expect(geminiService.bringToLife).not.toHaveBeenCalled();

        // 4. Trigger an intervention
        const interveneBtn = screen.getByText(/intervene/i);
        fireEvent.click(interveneBtn);

        // 5. Verify Gemini bridge is now called with the artifact content
        await waitFor(() => {
            expect(geminiService.bringToLife).toHaveBeenCalled();
            const callArgs = vi.mocked(geminiService.bringToLife).mock.calls[0];
            expect(callArgs[0]).toContain('"safety_score": 0.85');
        });
    });
});
