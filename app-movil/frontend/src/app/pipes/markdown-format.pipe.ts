import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'markdownFormat',
})
export class MarkdownFormatPipe implements PipeTransform {
  transform(value: string): string {
    if (!value) return '';

    value = value.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    value = value.replace(/\n/g, '<br>');

    return value;
  }
}
