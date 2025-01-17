const { Pool } = require('pg');

const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: {
        rejectUnauthorized: false
    }
});

async function getUserByDiscordId(discordId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'SELECT * FROM users WHERE discord_id = $1',
            [discordId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function createUser(discordId, email) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'INSERT INTO users (discord_id, email) VALUES ($1, $2) RETURNING *',
            [discordId, email]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function updateUserStripeCustomerId(userId, stripeCustomerId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'UPDATE users SET stripe_customer_id = $1 WHERE id = $2 RETURNING *',
            [stripeCustomerId, userId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function getSubscriptionByUserId(userId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'SELECT * FROM subscriptions WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1',
            [userId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function createSubscription(userId, stripeCustomerId, stripeSubscriptionId, planId, status) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            `INSERT INTO subscriptions 
             (user_id, stripe_customer_id, stripe_subscription_id, plan_id, status)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING *`,
            [userId, stripeCustomerId, stripeSubscriptionId, planId, status]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function updateSubscriptionStatus(subscriptionId, status) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'UPDATE subscriptions SET status = $1 WHERE id = $2 RETURNING *',
            [status, subscriptionId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function updateSubscriptionPlan(subscriptionId, planId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            'UPDATE subscriptions SET plan_id = $1 WHERE id = $2 RETURNING *',
            [planId, subscriptionId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function createPaymentRecord(userId, subscriptionId, amount, status, stripePaymentId) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            `INSERT INTO payments 
             (user_id, subscription_id, amount, status, stripe_payment_id)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING *`,
            [userId, subscriptionId, amount, status, stripePaymentId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

async function getPaymentHistory(userId, limit = 10) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            `SELECT * FROM payments 
             WHERE user_id = $1 
             ORDER BY created_at DESC 
             LIMIT $2`,
            [userId, limit]
        );
        return result.rows;
    } finally {
        client.release();
    }
}

async function updateSubscriptionDates(subscriptionId, currentPeriodStart, currentPeriodEnd) {
    const client = await pool.connect();
    try {
        const result = await client.query(
            `UPDATE subscriptions 
             SET current_period_start = $1, current_period_end = $2 
             WHERE id = $3 
             RETURNING *`,
            [currentPeriodStart, currentPeriodEnd, subscriptionId]
        );
        return result.rows[0];
    } finally {
        client.release();
    }
}

module.exports = {
    pool,
    getUserByDiscordId,
    createUser,
    updateUserStripeCustomerId,
    getSubscriptionByUserId,
    createSubscription,
    updateSubscriptionStatus,
    updateSubscriptionPlan,
    createPaymentRecord,
    getPaymentHistory,
    updateSubscriptionDates
}; 