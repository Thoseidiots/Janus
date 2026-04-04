import { describe, it, expect } from 'vitest';
import * as router from './aiServiceRouter';
import type { GroundingSource, Storyboard, VideoOperation } from './aiServiceRouter';

describe('aiServiceRouter', () => {
  it('exports all required functions', () => {
    expect(typeof router.generateText).toBe('function');
    expect(typeof router.generateChatResponse).toBe('function');
    expect(typeof router.generateWithGoogleSearch).toBe('function');
    expect(typeof router.generateWithGoogleMaps).toBe('function');
    expect(typeof router.analyzeImage).toBe('function');
    expect(typeof router.analyzeVideo).toBe('function');
    expect(typeof router.generateStoryboardFromFile).toBe('function');
    expect(typeof router.generateStoryboardFromUrl).toBe('function');
    expect(typeof router.generateImage).toBe('function');
    expect(typeof router.editImage).toBe('function');
    expect(typeof router.generateVideoFromText).toBe('function');
    expect(typeof router.generateVideoFromImage).toBe('function');
    expect(typeof router.pollVideoOperation).toBe('function');
    expect(typeof router.generateSpeech).toBe('function');
  });

  it('exports the requested types', () => {
    const source: GroundingSource = { title: 'test', uri: 'https://example.com' };
    const storyboard: Storyboard = { overallSummary: 'test', scenes: [] };
    const operation: VideoOperation = { id: 'op-1', status: 'done', resultUrl: 'https://example.com' };

    expect(source.title).toBe('test');
    expect(storyboard.scenes).toHaveLength(0);
    expect(operation.status).toBe('done');
  });

  it('silently ignores arbitrary model names for generateText', async () => {
    const text = await router.generateText('Hello world', 'any-model-name');
    expect(text).toContain('Simulated text generation');
  });

  it('does not mutate process.env.API_KEY while calling router functions', async () => {
    process.env.API_KEY = 'SHOULD_NOT_BE_USED';
    await router.generateText('Hello world');
    await router.generateChatResponse([], 'hi', false);
    expect(process.env.API_KEY).toBe('SHOULD_NOT_BE_USED');
  });
});
