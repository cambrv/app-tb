// src/app/services/typewriter.service.ts
import { Injectable, NgZone } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class TypewriterService {
  constructor(private zone: NgZone) {}

  // Versión original (para objetos {contentFull, contentRendered})
  animate(
    target: { contentFull: string; contentRendered: string; isTyping?: boolean },
    opts: { typingSpeedMs?: number; smartPauses?: boolean; onTick?: () => void } = {}
  ) {
    const typingSpeedMs = opts.typingSpeedMs ?? 12;
    const smartPauses = opts.smartPauses ?? true;

    target.isTyping = true;
    target.contentRendered = '';

    let i = 0;
    let cancelled = false;

    const step = () => {
      if (cancelled) return;
      if (i >= target.contentFull.length) {
        target.isTyping = false;
        opts.onTick?.();
        return;
      }

      const chunkSize = 1 + Math.floor(Math.random() * 2); // 1–2 chars
      const next = target.contentFull.slice(i, i + chunkSize);
      target.contentRendered += next;
      i += chunkSize;

      opts.onTick?.();

      let delay = typingSpeedMs;
      if (smartPauses) {
        const last = next[next.length - 1];
        if (/[.,;:!?]/.test(last)) delay += 120 + Math.random() * 120;
        if (/\n/.test(last)) delay += 80;
      }

      this.zone.runOutsideAngular(() => {
        setTimeout(() => this.zone.run(step), delay);
      });
    };

    step();
    return () => { cancelled = true; target.isTyping = false; };
  }

  // NUEVO: versión para tu ChatMsg {text, renderedText}
  animateMsg(
    msg: { text: string; renderedText: string; isTyping?: boolean },
    opts: { typingSpeedMs?: number; smartPauses?: boolean; onTick?: () => void } = {}
  ) {
    // Proxy que mapea a las claves que usa animate()
    const proxy = {
      get contentFull() { return msg.text; },
      set contentFull(v: string) { msg.text = v; },
      get contentRendered() { return msg.renderedText; },
      set contentRendered(v: string) { msg.renderedText = v; },
      get isTyping() { return msg.isTyping; },
      set isTyping(v: boolean | undefined) { msg.isTyping = v; },
    };

    return this.animate(proxy as any, opts);
  }
}
