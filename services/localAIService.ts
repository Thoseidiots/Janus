export type VideoOperation = {
  id: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  progress?: number;
  resultUrl?: string;
  error?: string;
};

export type { GroundingSource, Storyboard } from '../types';

export {
  generateText,
  generateChatResponse,
  generateWithGoogleSearch,
  generateWithGoogleMaps,
  analyzeImage,
  analyzeVideo,
  generateStoryboardFromFile,
  generateStoryboardFromUrl,
  generateImage,
  editImage,
  generateVideoFromText,
  generateVideoFromImage,
  pollVideoOperation,
  generateSpeech,
} from './aiService';
