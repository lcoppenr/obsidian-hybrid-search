/**
 * Every value of every Studio-service setting in src/config.ts.
 */
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, it } from 'vitest';
import { config, effectiveServiceSettings, validateServiceConfig } from '../src/config.js';

const NAMES = [
  'OBSIDIAN_UNREADABLE_VAULT',
  'OBSIDIAN_UNREADABLE_SUBTREE',
  'OBSIDIAN_MIN_SCAN_RATIO',
  'OBSIDIAN_RESCAN_MINUTES',
  'OBSIDIAN_MCP_TOKEN_FILE',
  'OBSIDIAN_TEXT_EXTENSIONS',
  'OBSIDIAN_HTML_MODE',
  'OBSIDIAN_TEXT_MAX_KB',
];

afterEach(() => {
  for (const name of NAMES) delete process.env[name];
});

function rejects(name: string, value: string): void {
  process.env[name] = value;
  assert.throws(() => validateServiceConfig(), new RegExp(name), `${name}=${value}`);
  delete process.env[name];
}

describe('service settings', () => {
  it('defaults', () => {
    validateServiceConfig();
    assert.deepEqual(effectiveServiceSettings(), {
      unreadable_vault: 'keep',
      unreadable_subtree: 'keep',
      min_scan_ratio: 0.5,
      rescan_minutes: 0,
      mcp_token_file: null,
      text_extensions: [],
      html_mode: 'text',
      text_max_kb: 512,
    });
  });

  it('UNREADABLE_VAULT and UNREADABLE_SUBTREE accept keep and purge, case-insensitively', () => {
    for (const value of ['keep', 'purge', 'PURGE']) {
      process.env.OBSIDIAN_UNREADABLE_VAULT = value;
      process.env.OBSIDIAN_UNREADABLE_SUBTREE = value;
      validateServiceConfig();
      assert.equal(config.unreadableVault, value.toLowerCase());
      assert.equal(config.unreadableSubtree, value.toLowerCase());
    }
    rejects('OBSIDIAN_UNREADABLE_VAULT', 'wipe');
    rejects('OBSIDIAN_UNREADABLE_SUBTREE', 'skip');
  });

  it('MIN_SCAN_RATIO accepts 0..1', () => {
    for (const value of ['0', '0.5', '1']) {
      process.env.OBSIDIAN_MIN_SCAN_RATIO = value;
      assert.equal(config.minScanRatio, Number(value));
    }
    for (const bad of ['-0.1', '1.5', 'half']) rejects('OBSIDIAN_MIN_SCAN_RATIO', bad);
  });

  it('RESCAN_MINUTES accepts 0 and positive numbers', () => {
    for (const value of ['0', '15', '0.5']) {
      process.env.OBSIDIAN_RESCAN_MINUTES = value;
      assert.equal(config.rescanMinutes, Number(value));
    }
    for (const bad of ['-1', 'hourly']) rejects('OBSIDIAN_RESCAN_MINUTES', bad);
  });

  it('MCP_TOKEN_FILE expands ~/', () => {
    process.env.OBSIDIAN_MCP_TOKEN_FILE = '~/Tools/ohs-service/token';
    assert.equal(config.mcpTokenFile, path.join(os.homedir(), 'Tools/ohs-service/token'));
    process.env.OBSIDIAN_MCP_TOKEN_FILE = '/etc/ohs/token';
    assert.equal(config.mcpTokenFile, '/etc/ohs/token');
    process.env.OBSIDIAN_MCP_TOKEN_FILE = '  ';
    assert.equal(config.mcpTokenFile, undefined);
  });

  it('TEXT_EXTENSIONS normalises case, dots, spaces and duplicates', () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS =
      'ps1,psm1,psd1,sh,zsh,py,bat,cmd,html,htm,txt,json,yaml,yml, .PS1 ';
    assert.deepEqual(config.textExtensions, [
      'bat',
      'cmd',
      'htm',
      'html',
      'json',
      'ps1',
      'psd1',
      'psm1',
      'py',
      'sh',
      'txt',
      'yaml',
      'yml',
      'zsh',
    ]);
    rejects('OBSIDIAN_TEXT_EXTENSIONS', 'md');
    rejects('OBSIDIAN_TEXT_EXTENSIONS', 'ps1,*.sh');
  });

  it('HTML_MODE accepts text and raw', () => {
    for (const value of ['text', 'raw']) {
      process.env.OBSIDIAN_HTML_MODE = value;
      assert.equal(config.htmlMode, value);
    }
    rejects('OBSIDIAN_HTML_MODE', 'markdown');
  });

  it('TEXT_MAX_KB accepts numbers >= 1', () => {
    process.env.OBSIDIAN_TEXT_MAX_KB = '1024';
    assert.equal(config.textMaxKb, 1024);
    for (const bad of ['0', '-5', 'big']) rejects('OBSIDIAN_TEXT_MAX_KB', bad);
  });
});
