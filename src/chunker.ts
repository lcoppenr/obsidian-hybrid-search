import { config } from './config.js'

export interface Chunk {
  text: string
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

interface Section {
  heading: string
  body: string
  text: string
}

export function splitBySections(content: string): Section[] {
  const lines = content.split('\n')
  const sections: Section[] = []
  let currentHeading = ''
  let currentBody: string[] = []

  const flush = () => {
    const body = currentBody.join('\n')
    if (body.trim().length >= config.chunkMinLength) {
      const text = currentHeading
        ? `${currentHeading}\n${body}`.trim()
        : body.trim()
      sections.push({ heading: currentHeading, body, text })
    }
    currentBody = []
  }

  for (const line of lines) {
    const isHeading = /^#{1,6}\s+/.test(line)
    if (isHeading) {
      flush()
      currentHeading = line
    } else {
      currentBody.push(line)
    }
  }
  flush()

  return sections
}

export function slidingWindow(text: string, contextLength: number, overlap: number): Chunk[] {
  const charSize = contextLength * 4
  const charStep = Math.max(contextLength - overlap, Math.ceil(contextLength / 2)) * 4
  const chunks: Chunk[] = []

  let start = 0
  while (start < text.length) {
    const end = Math.min(start + charSize, text.length)
    const chunk = text.slice(start, end).trim()
    if (chunk.length >= config.chunkMinLength) {
      chunks.push({ text: chunk })
    }
    if (end >= text.length) break
    start += charStep
  }

  return chunks.length > 0 ? chunks : [{ text: text.trim() }]
}

export function chunkNote(content: string, contextLength: number): Chunk[] {
  if (estimateTokens(content) <= contextLength) {
    return [{ text: content.trim() }]
  }

  const sections = splitBySections(content)

  if (sections.length <= 1) {
    return slidingWindow(content, contextLength, config.chunkOverlap)
  }

  const chunks: Chunk[] = []
  for (const section of sections) {
    if (section.body.trim().length < config.chunkMinLength) continue
    if (estimateTokens(section.text) <= contextLength) {
      chunks.push({ text: section.text })
    } else {
      chunks.push(...slidingWindow(section.text, contextLength, config.chunkOverlap))
    }
  }

  return chunks.length > 0 ? chunks : slidingWindow(content, contextLength, config.chunkOverlap)
}
