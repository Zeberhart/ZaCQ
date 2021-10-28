const prod = {
    url: "wss://handoff-server.herokuapp.com/client"
};

const dev = {
 url: 'wss://handoff-server.herokuapp.com/client'
};

export const config = process.env.NODE_ENV === 'development' ? dev : prod;