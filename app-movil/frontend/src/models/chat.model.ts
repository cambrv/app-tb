export interface ChatMessage {
  // id: string;
  // role: 'user' | 'assistant';
  contentFull: string;
  contentRendered: string;
  isTyping?: boolean;
}

export interface ChatMsg {
  isUser: boolean;
  text: string;
  renderedText: string;
  isTyping?: boolean;
}
