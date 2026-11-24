// Sample Motion Canvas easing on a shared clock for both aspect ratios.
import {easeInOutCubic, easeOutExpo} from '@motion-canvas/core/lib/tweening/timingFunctions';
const sample = curve => Array.from({length: 121}, (_, i) => curve(i / 120));
console.log(JSON.stringify({reveal: sample(easeOutExpo), transform: sample(easeInOutCubic)}));
