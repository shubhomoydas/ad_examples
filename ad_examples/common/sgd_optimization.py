import numpy as np
from .utils import matrix, logger, Timer


def get_num_batches(n, batch_size):
    return int(np.ceil(n * 1.0 / batch_size))


def get_sgd_batch(x, y, i, batch_size, shuffled_idxs=None):
    s = i * batch_size
    e = min(x.shape[0], (i + 1) * batch_size)
    if shuffled_idxs is None:
        idxs = np.arange(s, e)
    else:
        idxs = shuffled_idxs[np.arange(s, e)]
    return matrix(x[idxs, :], ncol=x.shape[1]), y[idxs]


def avg_loss_check(losses, epoch, n=20, eps=1e-6):
    if epoch < n + 1:
        return False
    avg1 = np.mean(losses[(epoch-1-n):(epoch-1)])
    avg2 = np.mean(losses[(epoch-n):(epoch)])
    if np.abs(avg1 - avg2) < eps:
        return True
    return False


def debug_log_sgd_losses(sgd_type, losses, epoch, n=20, timer=None):
    if False:
        # disable logging -- should be used in PRODUCTION
        return
    elif True:
        # minimal info
        logger.debug("[%s] epochs: %d; avg last %d losses:%f%s" %
                     (sgd_type, epoch, n, np.mean(losses[(epoch-min(n, epoch)):(epoch)]),
                      "" if timer is None else "; time: %f" % timer.elapsed()))
    else:
        # maximum info
        logger.debug("[%s] epochs: %d; avg last %d losses:%f\n%s\n%s" %
                     (sgd_type, epoch, n, np.mean(losses[(epoch-min(n, epoch)):(epoch)]),
                      str(list(losses[0:min(n, epoch)])),
                      str(list(losses[(epoch-min(n, epoch)):(epoch)]))))


def sgd(w0, x, y, f, grad, learning_rate=0.01,
        batch_size=100, max_epochs=1000, eps=1e-6, shuffle=False, rng=None):
    tm = Timer()
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = np.copy(w0)
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = np.copy(w0)
    loss_best = np.inf
    if n <= batch_size:
        shuffle = False  # no need to shuffle since all instances will be used up in one batch
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            if xi.shape[0] == 0:
                raise ValueError("Batch size of 0")
            g = grad(w, xi, yi)
            w -= learning_rate * g
            losses[i] = f(w, xi, yi)
            if False:
                g_norm = g.dot(g)
                if np.isnan(g_norm) or np.isinf(g_norm):
                    logger.debug("|grad|=%f, i=%d/%d, epoch:%d" % (g.dot(g), i+1, n_batches, epoch))
                    logger.debug("|w0|=%f" % w0.dot(w0))
                    raise ArithmeticError("grad is nan/inf in sgd")
        loss = np.mean(losses)
        if np.isnan(loss):
            logger.debug("loss is nan")
            logger.debug("|w|=%f" % w.dot(w))
            raise ArithmeticError("loss is nan in sgd")
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            np.copyto(w_best, w)
            loss_best = loss
        epoch += 1
        if loss < eps:
            break
    debug_log_sgd_losses("sgd", epoch_losses, epoch, n=20, timer=tm)
    # logger.debug("epochs: %d" % epoch)
    # logger.debug("net losses:")
    # logger.debug("epoch losses:\n%s" % str(epoch_losses[0:epoch]))
    # logger.debug("best loss: %f" % loss_best)
    return w_best


def sgdRMSProp(w0, x, y, f, grad, learning_rate=0.01,
               batch_size=100, max_epochs=1000, delta=1e-6, ro=0.9, eps=1e-6,
               shuffle=False, rng=None):
    tm = Timer()
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = np.copy(w0)
    r = np.zeros(len(w0), dtype=w0.dtype)  # gradient accumulation variable
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = np.copy(w0)
    loss_best = np.inf
    if n <= batch_size:
        # no need to shuffle since all instances will be used up in one batch
        shuffle = False
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    prev_loss = np.inf
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            g = grad(w, xi, yi)
            r[:] = ro * r + (1 - ro) * np.multiply(g, g)
            dw_scale = (learning_rate / (np.sqrt(delta + r)))
            dw = np.multiply(dw_scale, g)
            w[:] = w - dw
            losses[i] = f(w, xi, yi)
        loss = np.mean(losses)
        if np.isnan(loss):
            logger.debug("loss is nan")
            logger.debug("|w|=%f" % w.dot(w))
            raise ArithmeticError("loss is nan in sgd")
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            np.copyto(w_best, w)
            loss_best = loss
        epoch += 1
        if (loss < eps or np.abs(loss - prev_loss) < eps or
            avg_loss_check(epoch_losses, epoch, n=20, eps=eps)):
            break
        prev_loss = loss
    debug_log_sgd_losses("sgdRMSProp", epoch_losses, epoch, n=20, timer=tm)
    # logger.debug("epochs: %d" % epoch)
    # logger.debug("net losses:")
    # logger.debug("epoch losses:\n%s" % str(epoch_losses[0:epoch]))
    # logger.debug("best loss: %f" % loss_best)
    return w_best


def sgdMomentum(w0, x, y, f, grad, learning_rate=0.01,
                batch_size=100, max_epochs=1000,
                alpha=0.9, eps=1e-6,
                shuffle=False, rng=None):
    tm = Timer()
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = np.copy(w0)
    v = np.zeros(len(w0), dtype=w0.dtype)  # velocity
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = np.copy(w0)
    loss_best = np.inf
    if n <= batch_size:
        # no need to shuffle since all instances will be used up in one batch
        shuffle = False
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    prev_loss = np.inf
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            g = grad(w, xi, yi)
            v[:] = alpha * v - learning_rate * g
            w[:] = w + v
            losses[i] = f(w, xi, yi)
        loss = np.mean(losses)
        if np.isnan(loss):
            logger.debug("loss is nan")
            logger.debug("|w|=%f" % w.dot(w))
            raise ArithmeticError("loss is nan in sgd")
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            np.copyto(w_best, w)
            loss_best = loss
        epoch += 1
        if (loss < eps or np.abs(loss - prev_loss) < eps or
            avg_loss_check(epoch_losses, epoch, n=20, eps=eps)):
            break
        prev_loss = loss
    debug_log_sgd_losses("sgdMomentum", epoch_losses, epoch, n=20, timer=tm)
    # logger.debug("epochs: %d" % epoch)
    # logger.debug("net losses:")
    # logger.debug("epoch losses:\n%s" % str(epoch_losses[0:epoch]))
    # logger.debug("best loss: %f" % loss_best)
    return w_best


def sgdRMSPropNestorov(w0, x, y, f, grad, learning_rate=0.01,
                       batch_size=100, max_epochs=1000,
                       alpha=0.9, delta=1e-6, ro=0.9, eps=1e-6,
                       shuffle=False, rng=None):
    tm = Timer()
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = np.copy(w0)
    v = np.zeros(len(w0), dtype=w0.dtype)  # velocity
    r = np.zeros(len(w0), dtype=w0.dtype)  # gradient accumulation variable
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = np.copy(w0)
    loss_best = np.inf
    if n <= batch_size:
        # no need to shuffle since all instances will be used up in one batch
        shuffle = False
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    prev_loss = np.inf
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            tw = w + alpha * v
            g = grad(tw, xi, yi)
            r[:] = ro * r + (1 - ro) * np.multiply(g, g)
            dw_scale = (learning_rate / (np.sqrt(delta + r)))
            v = alpha * v - np.multiply(dw_scale, g)
            w[:] = w + v
            losses[i] = f(w, xi, yi)
        loss = np.mean(losses)
        if np.isnan(loss):
            logger.debug("loss is nan")
            logger.debug("|w|=%f" % w.dot(w))
            raise ArithmeticError("loss is nan in sgd")
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            np.copyto(w_best, w)
            loss_best = loss
        epoch += 1
        if (loss < eps or np.abs(loss - prev_loss) < eps or
            avg_loss_check(epoch_losses, epoch, n=20, eps=eps)):
            break
        prev_loss = loss
    debug_log_sgd_losses("sgdRMSPropNestorov", epoch_losses, epoch, n=20, timer=tm)
    # logger.debug("epochs: %d" % epoch)
    # logger.debug("net losses:")
    # logger.debug("epoch losses:\n%s" % str(epoch_losses[0:epoch]))
    # logger.debug("best loss: %f" % loss_best)
    return w_best


def sgdAdam(w0, x, y, f, grad, learning_rate=0.01,
            batch_size=100, max_epochs=1000, delta=1e-8,
            ro1=0.9, ro2=0.999, eps=1e-6,
            shuffle=False, rng=None):
    tm = Timer()
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = np.copy(w0)
    s = np.zeros(len(w0), dtype=w0.dtype)  # first moment variable
    s_hat = np.zeros(len(w0), dtype=w0.dtype)  # first moment corrected for bias
    r = np.zeros(len(w0), dtype=w0.dtype)  # second moment variable
    r_hat = np.zeros(len(w0), dtype=w0.dtype)  # second moment corrected for bias
    t = 0  # time step
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = np.copy(w0)
    loss_best = np.inf
    if n <= batch_size:
        # no need to shuffle since all instances will be used up in one batch
        shuffle = False
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    prev_loss = np.inf
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            g = grad(w, xi, yi)
            t += 1
            s[:] = ro1 * s + (1 - ro1) * g
            r[:] = ro2 * r + (1 - ro2) * np.multiply(g, g)
            # correct bias in first moment
            s_hat[:] = (1./(1 - ro1 ** t)) * s
            # correct bias in second moment
            r_hat[:] = (1./(1 - ro2 ** t)) * r
            dw_scale = (learning_rate / (np.sqrt(delta + r_hat)))
            dw = np.multiply(dw_scale, s_hat)
            w[:] = w - dw
            losses[i] = f(w, xi, yi)
        loss = np.mean(losses)
        if np.isnan(loss):
            logger.debug("loss is nan")
            logger.debug("|w|=%f" % w.dot(w))
            raise ArithmeticError("loss is nan in sgd")
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            np.copyto(w_best, w)
            loss_best = loss
        epoch += 1
        if (loss < eps or np.abs(loss - prev_loss) < eps or
            avg_loss_check(epoch_losses, epoch, n=20, eps=eps)):
            break
        prev_loss = loss
    debug_log_sgd_losses("sgdAdam", epoch_losses, epoch, n=20, timer=tm)
    # logger.debug("epochs: %d" % epoch)
    # logger.debug("net losses:")
    # logger.debug("epoch losses:\n%s" % str(epoch_losses[0:epoch]))
    # logger.debug("best loss: %f" % loss_best)
    return w_best

