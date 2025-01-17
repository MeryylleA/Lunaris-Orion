const PLANS = {
    premium: {
        id: 'premium',
        name: 'Premium',
        description: 'Full access to all premium features',
        price_id: process.env.STRIPE_PREMIUM_PRICE_ID,
        price: 10.00,
        features: [
            'Unlimited generations',
            'Access to all styles',
            'Maximum resolution',
            'Priority queue',
            'Priority support',
            'No watermark',
            'Early access to new features',
            'Custom negative prompts',
            'Advanced settings control'
        ],
        limits: {
            daily_generations: -1, // unlimited
            max_resolution: 256,
            concurrent_requests: 5
        },
        color: '#FFD700'
    }
};

const SUBSCRIPTION_STATUS = {
    active: {
        name: 'Active',
        color: '#2ecc71',
        emoji: '✅'
    },
    canceled: {
        name: 'Canceled',
        color: '#e74c3c',
        emoji: '❌'
    },
    past_due: {
        name: 'Payment Due',
        color: '#f1c40f',
        emoji: '⚠️'
    },
    incomplete: {
        name: 'Incomplete',
        color: '#95a5a6',
        emoji: '⏳'
    },
    incomplete_expired: {
        name: 'Expired',
        color: '#7f8c8d',
        emoji: '⌛'
    },
    trialing: {
        name: 'Trial Period',
        color: '#3498db',
        emoji: '🔄'
    },
    unpaid: {
        name: 'Unpaid',
        color: '#e67e22',
        emoji: '💰'
    }
};

module.exports = {
    PLANS,
    SUBSCRIPTION_STATUS,
    getPlan: (planId) => PLANS[planId] || null,
    getStatus: (status) => SUBSCRIPTION_STATUS[status] || SUBSCRIPTION_STATUS.incomplete,
    isValidPlan: (planId) => !!PLANS[planId]
}; 