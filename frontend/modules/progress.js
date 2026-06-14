/**
 * Progress overlay with elapsed-time tracker.
 */
import { formatBytes, formatElapsedTime, sleep } from './utils.js';

/** @type {number[]} */
let progressTimers = [];
/** @type {number|null} */
let predictStatusPollId = null;
/** @type {number|null} */
let outputSubStepTimer = null;
