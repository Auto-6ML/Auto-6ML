import torch.nn.functional as F
import torch
import torch.nn as nn

def train(model, embs, data, optimizer):
    optimizer.zero_grad()
    loss = F.cross_entropy(F.softmax(model(embs)[data.train_mask], dim=1), data.y[data.train_mask])
    # print("train loss: {}".format(loss))
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss

def val(model, embs, data):
    model.eval()
    logits = model(embs)
    pred = logits[data.val_mask].max(1)[1]
    acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    return acc

def test(model, embs, data):
    model.eval()
    logits, accs = model(embs), []
    pred = logits[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def train_model(model, name, embs, data, optimizer, patient = 100):
    model.train()
    best_acc = 0
    best_loss = 0
    for epoch in range(100000):
        loss_now = train(model, embs, data, optimizer)
        acc_now = val(model, embs, data)
        if acc_now > best_acc:
            best_acc = acc_now
            bad_counter = 0
            torch.save(model.state_dict(), 'best_model_{}.pkl'.format(name))
        else:
            bad_counter += 1
        if bad_counter == patient:
            break
    model.load_state_dict(torch.load('best_model_{}.pkl'.format(name)))
    return model

def train_s(model, data, optimizer):
    best_acc_val = 0
    best_acc = 0
    bad_counter = 0
    for epoch in range(100000):
        model.train()
        optimizer.zero_grad()
        loss_now = F.cross_entropy(F.softmax(model(data), dim = 1)[data.train_mask], data.y[data.train_mask])
        loss_now.backward()
        optimizer.step()

        model.eval()
        pre = model(data)
        pre_val = pre[data.val_mask].max(1)[1]
        pre_test = pre[data.test_mask].max(1)[1]
        acc = pre_val.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

        # print(acc_now)
        if acc > best_acc_val:
            best_acc_val = acc
            best_acc = pre_test.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            bad_counter = 0
            # print(epoch, best_acc_val)
        else:
            bad_counter += 1
        if bad_counter == 100:
            break
    print("student_1 test acc: {}".format(best_acc))

def train_student(model, name, data, teacher_pre, optimizer, clean, patient = 100):
    best_acc = 0
    train_mask = []
    for i in range(data.size(0)):
        if data.val_mask[i] == 0 and data.test_mask[i] == 0:
            train_mask.append(i)
    for epoch in range(100000):
        optimizer.zero_grad()
        # loss = train_s(model, data, teacher_pre, train_mask)
        loss = -(teacher_pre[train_mask] * F.log_softmax(model(data)[train_mask], dim=1)).mean()
        loss.backward(retain_graph=True)
        optimizer.step()
        model.eval()
        logits = model(data)
        pred = logits[data.val_mask].max(1)[1]
        acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        acc_now = acc

        if acc_now > best_acc:
            best_acc = acc_now
            best_loss = loss
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), 'best_model_{}.pkl'.format(name))
            # print(epoch, acc_now)
        else:
            bad_counter += 1
        if bad_counter == patient:
            break
    model.load_state_dict(torch.load('best_model_{}.pkl'.format(name)))
    model.eval()
    logits, accs = model(data), []
    pred = logits[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print("{} test acc: {}".format(name, acc))
    return model, acc, best_acc
