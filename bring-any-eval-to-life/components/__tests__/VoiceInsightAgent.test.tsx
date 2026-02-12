import { render, screen, fireEvent } from '@testing-library/react';
import { VoiceInsightAgent } from '../VoiceInsightAgent';
import { expect, test, vi } from 'vitest';
import React from 'react';

// Mock the ElevenLabs SDK
vi.mock('@elevenlabs/react', () => ({
    useConversation: () => ({
        startSession: vi.fn(),
        endSession: vi.fn(),
        sendUserMessage: vi.fn(),
    }),
}));

// Mock the ConversationBar to avoid complex UI testing
vi.mock('../ui/conversation-bar', () => ({
    ConversationBar: () => <div data-testid="conversation-bar">Conversation Bar</div>,
}));

test('renders the voice agent trigger button initially', () => {
    render(<VoiceInsightAgent />);
    const button = screen.getByRole('button', { name: /Voice Research Assistant/i });
    expect(button).toBeDefined();
});

test('expands the conversation bar when clicked', () => {
    render(<VoiceInsightAgent />);
    const button = screen.getByRole('button', { name: /Voice Research Assistant/i });
    fireEvent.click(button);

    const bar = screen.getByTestId('conversation-bar');
    expect(bar).toBeDefined();
});
