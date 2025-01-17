const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const paddleWidth = 10;
const paddleHeight = 60;
const ballSize = 10;
let leftPaddleY = canvas.height / 2 - paddleHeight / 2;
let rightPaddleY = canvas.height / 2 - paddleHeight / 2;
let ballX = canvas.width / 2;
let ballY = canvas.height / 2;
let ballSpeedX = 5;
let ballSpeedY = 5;

function draw() {
    // Clear canvas
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw paddles
    ctx.fillStyle = 'black';
    ctx.fillRect(0, leftPaddleY, paddleWidth, paddleHeight);
    ctx.fillRect(canvas.width - paddleWidth, rightPaddleY, paddleWidth, paddleHeight);

    // Draw ball
    ctx.beginPath();
    ctx.arc(ballX, ballY, ballSize, 0, Math.PI * 2);
    ctx.fillStyle = 'red';
    ctx.fill();
}

function update() {
    // Ball movement
    ballX += ballSpeedX;
    ballY += ballSpeedY;

    // Ball collision with top/bottom
    if (ballY - ballSize < 0 || ballY + ballSize > canvas.height) {
        ballSpeedY = -ballSpeedY;
    }

    // Ball collision with paddles
    if (ballX - ballSize < paddleWidth) {
        if (ballY > leftPaddleY && ballY < leftPaddleY + paddleHeight) {
            ballSpeedX = -ballSpeedX;
        } else if (ballX < 0) {
            ballX = canvas.width / 2;
            ballY = canvas.height / 2;
        }
    }
    if (ballX + ballSize > canvas.width - paddleWidth) {
        if (ballY > rightPaddleY && ballY < rightPaddleY + paddleHeight) {
            ballSpeedX = -ballSpeedX;
        } else if (ballX > canvas.width) {
            ballX = canvas.width / 2;
            ballY = canvas.height / 2;
        }
    }

    // AI for right paddle
    if (ballSpeedX > 0) {
        if (rightPaddleY + paddleHeight / 2 < ballY) {
            rightPaddleY += 3;
        } else if (rightPaddleY + paddleHeight / 2 > ballY) {
            rightPaddleY -= 3;
        }
    }

    // Keep paddles within canvas
    leftPaddleY = Math.max(0, Math.min(leftPaddleY, canvas.height - paddleHeight));
    rightPaddleY = Math.max(0, Math.min(rightPaddleY, canvas.height - paddleHeight));
}

function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

canvas.addEventListener('mousemove', (e) => {
    leftPaddleY = e.clientY - canvas.offsetTop - paddleHeight / 2;
});

gameLoop();
