import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ResultSelector } from '../components/ResultSelector';
import { EvaluationArtifact } from '../services/github';

const mockArtifacts: EvaluationArtifact[] = [
    {
        id: '1',
        name: 'safety-eval-1.json',
        path: 'results/safety-eval-1.json',
        sha: 'abcdef123',
        url: '',
        html_url: '',
        download_url: '',
        content: {
            safety_score: 0.9,
            status: 'SAFE',
            summary: 'Clinical scenario test'
        }
    },
    {
        id: '2',
        name: 'unsafe-eval.json',
        path: 'results/unsafe-eval.json',
        sha: '987654321',
        url: '',
        html_url: '',
        download_url: '',
        content: {
            safety_score: 0.2,
            status: 'UNSAFE',
            summary: 'Failed protocol test'
        }
    }
];

describe('ResultSelector Component', () => {
    it('renders loading state', () => {
        const { container } = render(<ResultSelector artifacts={[]} onSelect={() => { }} isLoading={true} />);
        expect(container.querySelector('.animate-pulse')).toBeDefined();
    });

    it('renders empty state', () => {
        render(<ResultSelector artifacts={[]} onSelect={() => { }} />);
        expect(screen.getByText(/No artifacts found/i)).toBeDefined();
    });

    it('renders artifacts correctly', () => {
        render(<ResultSelector artifacts={mockArtifacts} onSelect={() => { }} />);

        expect(screen.getByText(/safety eval 1/i)).toBeDefined();
        expect(screen.getByText(/unsafe eval/i)).toBeDefined();
        expect(screen.getByText('90')).toBeDefined(); // Safety score
        expect(screen.getByText('20')).toBeDefined(); // Safety score
    });

    it('calls onSelect when a card is clicked', () => {
        const onSelect = vi.fn();
        render(<ResultSelector artifacts={mockArtifacts} onSelect={onSelect} />);

        const card = screen.getByText(/safety eval 1/i).closest('div');
        if (card) fireEvent.click(card);

        expect(onSelect).toHaveBeenCalledWith(mockArtifacts[0]);
    });
});
