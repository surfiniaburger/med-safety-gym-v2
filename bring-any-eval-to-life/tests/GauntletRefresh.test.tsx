import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
import * as githubService from '../services/github';
import * as geminiService from '../services/gemini';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fetchHubArtifacts } from '../services/hub';

// Mock the services
vi.mock('../services/github');
vi.mock('../services/gemini');
vi.mock('../services/hub');

vi.mock('../components/Gauntlet/GauntletView', () => ({
    GauntletView: vi.fn(({ onIntervene, activeStepIndex }) => (
        <div data-testid="gauntlet-view">
            Gauntlet View Mock - Node {activeStepIndex}
            <button onClick={() => onIntervene(0)}>Intervene Node 0</button>
            <button onClick={() => onIntervene(1)}>Intervene Node 1</button>
        </div>
    ))
}));

vi.mock('../components/LivePreview', () => ({
    LivePreview: ({ creation, onSolveNode, isFocused }) => (
        isFocused ? (
            <div data-testid="live-preview">
                {creation ? `Simulation: ${creation.html}` : 'No Creation'}
                <button onClick={onSolveNode}>Complete</button>
            </div>
        ) : null
    )
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

describe('Gauntlet Simulation Refresh', () => {
    beforeEach(() => {
        vi.resetAllMocks();
        vi.mocked(fetchHubArtifacts).mockResolvedValue([]);
    });

    it('refreshes the simulation when moving to a new intervention node', async () => {
        const mockArtifacts: githubService.EvaluationArtifact[] = [
            {
                id: '1',
                name: 'test-artifact.json',
                path: 'results/test-artifact.json',
                sha: 'sha1',
                url: '', html_url: '', download_url: '',
                content: {
                    safety_score: 0.5,
                    status: 'UNSAFE',
                    summary: 'Multi-failure test',
                    results: [{
                        results: [{
                            summary: { rewards: [-10, -20] },
                            detailed_results: [
                                { reward: -10, metrics: { hallucination: true } },
                                { reward: -20, metrics: { format_error: true } }
                            ]
                        }]
                    }]
                }
            }
        ];

        vi.mocked(githubService.fetchEvaluationArtifacts).mockResolvedValue(mockArtifacts);
        
        // Mock Gemini to return different content for each call
        vi.mocked(geminiService.bringToLife)
            .mockResolvedValueOnce('<html>Node 0 Simulation</html>')
            .mockResolvedValueOnce('<html>Node 1 Simulation</html>');

        render(<App />, { wrapper });

        // 1. Select artifact
        await waitFor(() => expect(screen.getByText(/test artifact/i)).toBeDefined());
        const card = screen.getByText(/test artifact/i).closest('div');
        if (card) fireEvent.click(card);

        // 2. Trigger intervention on Node 0
        await waitFor(() => expect(screen.getByTestId('gauntlet-view')).toBeDefined());
        fireEvent.click(screen.getByText(/Intervene Node 0/i));

        // 3. Verify Node 0 simulation is shown
        await waitFor(() => {
            expect(screen.getByText(/Simulation: <html>Node 0 Simulation<\/html>/i)).toBeDefined();
        });
        expect(geminiService.bringToLife).toHaveBeenCalledTimes(1);

        // 4. Complete Node 0
        fireEvent.click(screen.getByText(/Complete/i));

        // 5. Verify we are back in Gauntlet
        await waitFor(() => {
            expect(screen.queryByTestId('live-preview')).toBeNull();
        });

        // 6. Trigger intervention on Node 1
        fireEvent.click(screen.getByText(/Intervene Node 1/i));

        // 7. Verify Node 1 simulation is shown (THIS IS WHERE IT SHOULD FAIL IF BUG EXISTS)
        await waitFor(() => {
            expect(screen.getByText(/Simulation: <html>Node 1 Simulation<\/html>/i)).toBeDefined();
        });
        expect(geminiService.bringToLife).toHaveBeenCalledTimes(2);
    });
});
