
export async function getVideoInputs() {
	if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
	  console.log('enumerateDevices() not supported.');
	  return [];
	}

	const devices = await navigator.mediaDevices.enumerateDevices();
	return devices.filter(device => device.kind === 'videoinput');
}

export async function setupCamera(deviceId) {
	if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
		throw new Error(
			'Browser API navigator.mediaDevices.getUserMedia not available');
	}

	const videoConfig = {
		audio: false,
		video: {
			deviceId,
		  	width: 1280,
		  	height: 720,
			frameRate: { ideal: 10, max: 15 },
		}
	};

	if (!deviceId) {
		delete videoConfig.video.deviceId;
	}

	const stream = await navigator.mediaDevices.getUserMedia(videoConfig);
	const video = document.querySelector('#video');

	video.srcObject = stream;
	video.onloadedmetadata = () => {
		video.play();
	};
}

export async function setupCameraApp() {
    const canvas = document.querySelector('#output');
	const cameras = await getVideoInputs();

	let deviceId;
	const cameraSelect = document.querySelector('#camera-select');
	for (let i = 0; i < cameras.length; i++) {
		const camera = cameras[i];
		const option = document.createElement('option');
		option.value = camera.deviceId;
		option.textContent = camera.label;
		cameraSelect.appendChild(option);

		if(i === 0) {
			deviceId = camera.deviceId;
		}
	}

	cameraSelect.addEventListener('change', () => setupCamera(cameraSelect.value));

	await setupCamera(deviceId);
}
