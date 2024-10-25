/** @module session */
import { isEmpty } from './helpers.js';

/**
 * A wrapper for the browser's local storage, session storage, and cookies.
 *
 * @class
 */
class SessionStorage {
    /**
     * @param {Storage} driver - The storage driver to use.
     * @constructor
     */
    constructor(driver) {
        this.driver = driver;
        this.scopes = {};
    }

    /**
     * Get a scoped storage object.
     *
     * @param {string} key - The key to use for the scope.
     * @param {number} ttl - The time-to-live for the scope.
     * @returns {ScopedStorage} The scoped storage object.
     */
    getScope(key, ttl) {
        if (!this.scopes.hasOwnProperty(key)) {
            this.scopes[key] = new ScopedStorage(key, this.driver, ttl);
        }
        return this.scopes[key];
    }

    /**
     * Gets all items from all scopes.
     *
     * @returns {Object} An object containing all items from all scopes.
     */
    getAll() {
        return Object.getOwnPropertyNames(this.scopes).reduce((acc, item) => {
            acc[item] = storage.getScope(item).getAll();
            return acc;
        }, {});
    }

    /**
     * Clears all items from all scopes.
     */
    clear() {
        for (let scope in this.scopes) {
            this.getScope(scope).clear();
        }
    }
}

/**
 * A scoped storage object.
 *
 * @class
 */
class ScopedStorage {
    /**
     * @param {string} scope - The scope to use.
     * @param {Storage} driver - The storage driver to use.
     * @param {number} ttl - The time-to-live for the scope.
     * @constructor
     */
    constructor(scope, driver, ttl) {
        this.scope = scope;
        this.driver = driver;
        this.ttl = ttl;
        if (this.ttl === undefined) this.ttl = 60 * 60 * 1000; // 1 hour
    }

    /**
     * Gets all items from the scope.
     *
     * @returns {Object} An object containing all items from the scope.
     */
    getAll() {
        return this.keys().reduce((acc, item) => {
            acc[item] = storage.getItem(item);
            return acc;
        }, {});
    }

    /**
     * Sets an item in the scope.
     *
     * @param {string} key - The key to use.
     * @param {any} value - The value to set.
     */
    setItem(key, value) {
        let scopedKey = `${this.scope}-${key}`,
            scopedExpirationKey = `${this.scope}-${key}-expiration`;

        this.driver.setItem(scopedKey, JSON.stringify(value));
        this.driver.setItem(
            scopedExpirationKey,
            new Date().getTime() + this.ttl
        );
    }

    /**
     * Gets an item from the scope.
     *
     * @param {string} key - The key to get.
     * @returns {any} The value of the key.
     */
    getItem(key) {
        let scopedKey = `${this.scope}-${key}`,
            scopedExpirationKey = `${this.scope}-${key}-expiration`,
            response = this.driver.getItem(scopedKey),
            expirationResponse = this.driver.getItem(scopedExpirationKey);

        if (response === undefined || response === 'undefined')
            return undefined;
        if (response === null || response === 'null') return null;

        response = JSON.parse(response);

        if (
            !isEmpty(expirationResponse) &&
            expirationResponse <= new Date().getTime()
        ) {
            this.removeItem(scopedKey);
            return null;
        }
        return response;
    }

    /**
     * Gets all keys from the scope.
     *
     * @returns {Array} An array containing all keys from the scope.
     */
    keys() {
        let scope = `${this.scope}-`, theseKeys;
        if (this.driver.keys !== undefined) {
            theseKeys = this.driver.keys();
        } else {
            theseKeys = Object.getOwnPropertyNames(this.driver);
        }

        return theseKeys
            .filter((key) => key.startsWith(scope))
            .map((key) => key.substring(scope.length))
            .filter((key) => key != 'expiration');
    }

    /**
     * Removes an item from the scope.
     *
     * @param {string} key - The key to remove
     */
    removeItem(key) {
        let scopedKey = `${this.scope}-${key}`;
        return this.driver.removeItem(scopedKey);
    }

    /**
     * Removes all items from the scope beginning with a prefix.
     *
     * @param {string} prefix - The prefix to remove.
     */
    removePrefix(prefix) {
        for (let key of this.keys()) {
            if (key.startsWith(prefix)) {
                this.removeItem(key);
            }
        }
    }

    /**
     * Clears all items from the scope.
     */
    clear() {
        for (let key of this.keys()) {
            this.removeItem(key);
        }
        this.setItem('expiration', {});
        return this.driver.clear();
    }

    /**
     * Gets the key at an index.
     * @param {number} index - The index to get.
     * @returns {string} The key at the index.
     */
    key(index) {
        return this.keys()[index];
    }
}

/**
 * A storage object using a cookie backend.
 *
 * @class
 */
class CookieStorage {
    /**
     * @constructor
     */
    constructor() {
        this.expiration = new Date();
        this.expiration.setTime(
            this.expiration.getTime() + 30 * 24 * 60 * 60 * 1000
        );
    }

    /**
     * Gets all keys from the cookie storage.
     *
     * @returns {Array} An array containing all keys from the cookie storage.
     */
    keys() {
        let cookies = decodeURIComponent(document.cookie).split(';'),
            cookieName,
            cookieNames = [];

        for (let cookie of cookies) {
            cookieName = cookie.split('=')[0];
            while (cookieName.charAt(0) == ' ') {
                cookieName = cookieName.substring(1);
            }
            cookieNames.push(cookieName);
        }
        return cookieNames;
    }

    /**
     * Gets an item from the cookie storage.
     *
     * @param {string} key - The key to get.
     * @returns {any} The value of the key. Null if the key doesn't exist.
     */
    getItem(key) {
        let cookies = decodeURIComponent(document.cookie).split(';');
        for (let cookie of cookies) {
            while (cookie.charAt(0) == ' ') {
                cookie = cookie.substring(1);
            }
            if (cookie.startsWith(`${key}=`)) {
                return JSON.parse(cookie.substring(key.length + 1));
            }
        }
        return null;
    }

    /**
     * Sets an item in the cookie storage.
     *
     * @param {string} key - The key to set.
     * @param {any} value - The value to set.
     * @param {Date} expiration - The expiration date of the cookie.
     */
    setItem(key, value, expiration) {
        if (expiration === undefined) {
            expiration = this.expiration;
        }
        document.cookie = [
            `${key}=${JSON.stringify(value)}`,
            `expires=${expiration.toUTCString()}`,
            "path=/",
            ""
        ].join(";");
    }

    /**
     * Removes an item from the cookie storage.
     *
     * @param {string} key - The key to remove.
     */
    removeItem(key) {
        let expiration = new Date();
        expiration.setTime(0);
        this.setItem(key, '', expiration);
    }

    /**
     * Removes all items from the cookie storage.
     */
    clear() {
        for (let cookieName of this.keys()) {
            this.removeItem(cookieName);
        }
    }

    /**
     * Gets the key at an index.
     * @param {number} index - The index to get.
     * @returns {string} The key at the index.
     */
    key(index) {
        return this.keys()[index];
    }
}

/**
 * A storage object using a memory backend.
 * Essentially a wrapper for an object, only used for compatibility.
 * @class
 */
class MemoryStorage {
    /**
     * @constructor
     */
    constructor() {
        this.storage = {};
    }

    /**
     * Gets all keys from the memory storage.
     *
     * @returns {Array} An array containing all keys from the memory storage.
     */
    keys() {
        return Object.getOwnPropertyNames(this.storage);
    }

    /**
     * Sets an item in the memory storage.
     *
     * @param {string} key - The key to set.
     * @param {any} value - The value to set.
     */
    setItem(key, value) {
        this.storage[key] = value;
    }

    /**
     * Gets an item from the memory storage.
     *
     * @param {string} key - The key to get.
     * @returns {any} The value of the key. Null if the key doesn't exist.
     */
    getItem(key) {
        if (this.storage.hasOwnProperty(key)) {
            return this.storage[key];
        }
        return null;
    }

    /**
     * Removes an item from the memory storage.
     *
     * @param {string} key - The key to remove.
     */
    removeItem(key) {
        delete this.storage[key];
    }


    /**
     * Removes all items from the memory storage beginning with a prefix.
     *
     * @param {string} prefix - The prefix to remove.
     */
    removePrefix(prefix) {
        for (let key of this.keys()) {
            if (key.startsWith(prefix)) {
                this.removeItem(key);
            }
        }
    }

    /**
     * Clears all items from the memory storage.
     */
    clear() {
        for (let key of this.keys) {
            delete this.storage[key];
        }
    }

    /**
     * Gets the key at an index.
     *
     * @param {number} index - The index to get.
     * @returns {string} The key at the index.
     */
    key(index) {
        return this.keys()[index];
    }
}

/**
 * Gets the best storage driver available.
 * @returns {Storage} The storage driver.
 */
let getDriver = function () {
    try {
        if (typeof Storage !== undefined) {
            return window.localStorage;
        }
    } catch (e) {
        console.error("Couldn't get local storage, defaulting to cookies.", e);
        return new CookieStorage();
    }
};

export { SessionStorage, MemoryStorage, CookieStorage, ScopedStorage };
export let Session = new SessionStorage(getDriver()); // Export one instance of the session storage.
